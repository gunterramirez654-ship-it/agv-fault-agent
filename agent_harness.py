#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_harness.py —— LLM 调用可靠性外壳（Agent Harness）
================================================================================

定位
--------------------------------------------------------------------------------
本模块不实现任何分析逻辑，只在「调用 LLM」这件事外面加一层工程保障。
它是 agv_agent.AgvRiskAgent 的外壳（wrapper），不是它的替代品。

    业务入口（test_vector_store.py）
              ↓
    ┌─────────────────────────────────────────┐
    │  AgentHarness                           │  ← 本模块
    │   ① 输出安全校验（硬约束）               │
    │   ② 失败重试（指数退避）                 │
    │   ③ 降级兜底                             │
    │   ④ 指标统计（耗时/token/成功率）        │
    └─────────────────────────────────────────┘
              ↓
    AgvRiskAgent.analyze()   ← 只负责拼 prompt 调模型

--------------------------------------------------------------------------
设计原则
--------------------------------------------------------------------------
1) 【安全边界不变】
   风险等级与控制动作仍由规则系统决定。本模块只做「校验模型输出有没有越界」，
   绝不参与决策，也不提供任何修改 rule_risk / rule_action 的接口。

2) 【分层不变】
   vector_store.py  → 召回
   agv_agent.py     → LLM 调用
   agent_harness.py → 可靠性外壳（本模块）
   simple_logger.py → 日志

3) 【对外接口极简】
   上层只需把 agent.analyze(...) 换成 harness.analyze(...)，其余代码不动。

--------------------------------------------------------------------------
四层保障详解
--------------------------------------------------------------------------
① 输出安全校验
   system prompt 里的约束是「软约束」——模型可能不遵守。本模块在模型返回后
   做「硬校验」：扫描报告中的控制类违规措辞（如"建议停机""可以恢复运行"），
   命中即判定输出不合格。同时校验是否含三段式结构、是否引用了规则结论。

② 失败重试
   接口超时、网络抖动、返回空内容等可恢复错误，按指数退避重试。
   默认重试 2 次（共 3 次尝试）。

③ 降级兜底
   重试耗尽或输出校验不通过时，返回安全的三段式兜底报告（原样引用规则结论），
   保证业务链路永远拿得到一份可读报告，不中断 AGV 主流程。

④ 指标统计
   记录每次调用的耗时、重试次数、token 用量、报告字数、校验结果、
   最终状态（成功/降级），供上层写入日志、做成本分析与可用性统计。

--------------------------------------------------------------------------
【策略归属】输出预算由本模块统一管控
--------------------------------------------------------------------------
   单次响应允许多长，是【调用策略】而不是 prompt 的一部分，因此归 Harness 管：
       · max_output_tokens : 硬上限，注入给 agent 作为 API 的 max_tokens
       · max_report_chars  : 软阈值，报告超长时记录告警（不判失败，避免无谓重试）

   这样调整输出预算只需改 Harness 的构造参数，不必改 agv_agent.py。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agv_agent import AgvRiskAgent

# =========================================================================== #
# 一、输出安全校验规则
# =========================================================================== #
# 说明：这些词一旦出现在报告里，说明模型越界输出了控制指令或试图改动安全结论。
#      规则采用「子串匹配」，因此每个词都要写得足够具体，避免误伤正常表述。
#      例如「建议停机」会命中，但正常的「停机记录」不会（不含"建议"前缀）。
FORBIDDEN_PATTERNS: Sequence[str] = (
    # —— 越界输出控制指令 ——
    "建议停机",
    "建议立即停机",
    "建议断电",
    "建议立即断电",
    "可以恢复运行",
    "可恢复运行",
    "无需停机",
    "不必停机",
    "建议复位",
    "建议送电",
    "建议启动",
    # —— 试图改动/质疑安全结论 ——
    "建议降级",
    "建议升级",
    "建议提高风险",
    "建议降低风险",
    "建议修改风险",
    "建议修改控制",
    "风险等级应改为",
    "动作应改为",
    "判定有误",
    "判定过重",
    "判定过轻",
    "过度反应",
)

# 报告必须包含的三段结构（缺任一即视为格式不合格）
REQUIRED_SECTIONS: Sequence[str] = ("一、匹配评估", "二、诱因推测", "三、排查建议")

# 提示模型必须原样引用规则结论的标记
RULE_CITATION_HINT = "规则系统判定"


# =========================================================================== #
# 二、数据结构
# =========================================================================== #
@dataclass
class ValidationResult:
    """输出安全校验结果。"""

    passed: bool
    violations: List[str] = field(default_factory=list)   # 命中的违规措辞
    missing_sections: List[str] = field(default_factory=list)  # 缺失的段落
    cites_rule: bool = False                             # 是否引用了规则结论
    report_chars: int = 0                                # 报告字数（观测用）
    too_long: bool = False                               # 是否超过软阈值（不判失败）

    @property
    def summary(self) -> str:
        """一句话描述校验结果，便于日志与打印。"""
        if self.passed:
            return "校验通过"
        problems: List[str] = []
        if self.violations:
            problems.append(f"违规措辞 {self.violations}")
        if self.missing_sections:
            problems.append(f"缺失段落 {self.missing_sections}")
        if not problems and self.too_long:
            # 超长不判失败，仅在摘要中提示，便于观察输出预算是否合理
            return f"校验通过（报告偏长 {self.report_chars} 字）"
        return "；".join(problems)

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典（写入日志用）。"""
        return {
            "passed": self.passed,
            "violations": self.violations,
            "missing_sections": self.missing_sections,
            "cites_rule": self.cites_rule,
            "report_chars": self.report_chars,
            "too_long": self.too_long,
        }


@dataclass
class HarnessResult:
    """Harness 一次调用的完整结果。

    上层通常只关心 report；其余字段用于日志、监控与成本统计。
    """

    report: str                      # 最终报告（真实报告或兜底报告）
    status: str                      # success / degraded
    attempts: int                    # 实际尝试次数
    cost_ms: float                   # 总耗时（含重试）
    validation: ValidationResult     # 最后一次校验结果
    prompt_tokens: int = 0           # 输入 token（模型返回时才有）
    completion_tokens: int = 0       # 输出 token
    total_tokens: int = 0
    report_chars: int = 0            # 报告字数（观测输出预算是否合理）
    error: Optional[str] = None      # 降级原因（success 时为 None）

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典（写入日志用）。"""
        return {
            "status": self.status,
            "attempts": self.attempts,
            "cost_ms": round(self.cost_ms, 1),
            "validation": self.validation.to_dict(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "report_chars": self.report_chars,
            "error": self.error,
        }


@dataclass
class HarnessStats:
    """累积统计，用于回答「这套 LLM 链路可靠性如何、成本多少」。"""

    total_calls: int = 0
    success_calls: int = 0
    degraded_calls: int = 0
    total_attempts: int = 0
    total_cost_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)

    def record(self, result: HarnessResult) -> None:
        """记录一次调用结果。"""
        self.total_calls += 1
        self.total_attempts += result.attempts
        self.total_cost_ms += result.cost_ms
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens

        if result.is_success:
            self.success_calls += 1
        else:
            self.degraded_calls += 1
            reason = result.error or "unknown"
            # 归类原因，便于统计「主要因为什么降级」
            key = "超时/网络异常" if "timeout" in reason.lower() or "connection" in reason.lower() \
                else ("输出校验不通过" if "校验" in reason else "其他")
            self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1

    @property
    def success_rate(self) -> float:
        """调用成功率（0~1）。"""
        return self.success_calls / self.total_calls if self.total_calls else 0.0

    @property
    def avg_cost_ms(self) -> float:
        """平均耗时（毫秒）。"""
        return self.total_cost_ms / self.total_calls if self.total_calls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典。"""
        return {
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "degraded_calls": self.degraded_calls,
            "success_rate": round(self.success_rate, 4),
            "total_attempts": self.total_attempts,
            "avg_cost_ms": round(self.avg_cost_ms, 1),
            "total_cost_ms": round(self.total_cost_ms, 1),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "reject_reasons": dict(self.reject_reasons),
        }


# =========================================================================== #
# 三、输出安全校验
# =========================================================================== #
def validate_report(report: str, max_chars: int = 800) -> ValidationResult:
    """对 LLM 报告做输出安全校验（硬约束）。

    校验三项：
        1) 是否含控制类违规措辞 —— 命中即不合格
        2) 是否含三段式结构     —— 缺任一即不合格
        3) 是否引用了规则结论   —— 仅记录，不作为合格条件
           （因为兜底报告本身就会引用，而正常报告也不强制要求引用）

    参数：
        report    : LLM 返回的报告文本
        max_chars : 报告字数软阈值，超过时只标记 too_long（不判失败）

    返回：
        ValidationResult
    """
    text = str(report or "")

    # ---- 1) 违规措辞扫描 ----
    violations = [p for p in FORBIDDEN_PATTERNS if p in text]

    # ---- 2) 三段结构检查 ----
    missing = [s for s in REQUIRED_SECTIONS if s not in text]

    # ---- 3) 规则结论引用检查（仅记录）----
    cites_rule = RULE_CITATION_HINT in text or "原样引用" in text

    report_chars = len(text.strip())
    return ValidationResult(
        passed=(not violations and not missing),
        violations=violations,
        missing_sections=missing,
        cites_rule=cites_rule,
        report_chars=report_chars,
        too_long=(max_chars > 0 and report_chars > max_chars),
    )


def _build_fallback_report(report: str, rule_risk: str, rule_action: str) -> str:
    """构造降级兜底报告。

    当重试耗尽、或模型输出未通过安全校验时使用。
    兜底报告原样引用规则系统结论，且不含任何控制类措辞。

    参数：
        report      : 模型最后一次的原始输出（作为参考附在后面，便于排查）
        rule_risk   : 规则系统判定的风险等级（只读）
        rule_action : 规则系统判定的控制动作（只读）

    返回：
        三段式兜底报告文本
    """
    blocked = str(report or "").strip()
    detail = ""
    if blocked:
        # 保留被拦截内容的摘要，但明确标注为「已拦截、不可作为处置依据」
        snippet = re.sub(r"\s+", " ", blocked)[:200]
        detail = f"\n（说明：模型原始输出未通过安全校验，已拦截。原文摘要：{snippet}……）"

    return (
        f"一、匹配评估：模型输出未通过安全校验，本次未能提供匹配度分析，"
        f"请以规则系统结论为准。\n"
        f"二、诱因推测：为保障安全，未采用未经校验的模型推测内容。\n"
        f"三、排查建议：1. 严格遵守规则系统判定的风险等级：{rule_risk}。"
        f"2. 严格执行规则系统判定的控制动作：{rule_action}。"
        f"3. 人工查阅本次事件对应的历史故障记录，核对现场实际状态。"
        f"4. 如需智能分析结论，请重新发起本次分析。{detail}"
    )


# =========================================================================== #
# 四、Harness 主体
# =========================================================================== #
class AgentHarness:
    """LLM 调用可靠性外壳。

    使用方式（对外接口与 AgvRiskAgent 保持一致，便于替换）：

        harness = AgentHarness()
        result = harness.analyze(
            query_text     = "现场事件文本",
            search_results = [...],
            rule_risk      = "高风险",
            rule_action    = "AGV立即暂停，人工确认现场",
        )
        print(result.report)          # 最终报告
        print(result.to_dict())       # 指标（写日志用）

    与直接调用 AgvRiskAgent 的区别：
        · 返回 HarnessResult（含报告 + 指标），而不是裸字符串；
        · 自动重试、安全校验、降级兜底；
        · 累积统计可通过 stats 属性读取。

    安全说明：
        本类不持有、不返回、也不修改任何控制决策。rule_risk 与 rule_action
        仅作为只读文本透传，用于提示词与兜底报告。
    """

    def __init__(
        self,
        agent: Optional[AgvRiskAgent] = None,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        max_delay: float = 8.0,
        max_output_tokens: int = 600,
        max_report_chars: int = 800,
    ) -> None:
        """初始化。

        参数：
            agent            : 被包装的 LLM 组件；不传时内部创建 AgvRiskAgent
            max_retries      : 失败后的最大重试次数（默认 2，即最多尝试 3 次）
            retry_base_delay : 首次重试等待秒数，之后指数退避（1s → 2s → 4s）
            max_delay        : 单次重试等待上限，避免等待过久影响现场时效
            max_output_tokens: 单次响应输出 token 硬上限（注入 agent 的 max_tokens）
            max_report_chars : 报告字数软阈值，超长时记录告警但不判失败
        """
        self.agent = agent or AgvRiskAgent()
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.max_delay = max(0.0, float(max_delay))
        # —— 输出预算：由本模块统一管控，注入给 agent 使用 ——
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.max_report_chars = max(0, int(max_report_chars))
        self._apply_output_budget()
        self.stats = HarnessStats()

    def _apply_output_budget(self) -> None:
        """把输出预算注入 agent。

        说明：agv_agent 把 max_tokens 暴露为实例属性 max_output_tokens，
             这里覆盖它即可生效，无需改 agent 的调用签名。
             若 agent 不支持该属性（自定义实现），静默跳过，不影响主流程。
        """
        try:
            if hasattr(self.agent, "max_output_tokens"):
                self.agent.max_output_tokens = self.max_output_tokens
        except Exception:  # noqa: BLE001  预算注入失败不应影响调用
            pass

    # ------------------------------------------------------------------ #
    # 对外唯一接口
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        query_text: str,
        search_results: list,
        rule_risk: str,
        rule_action: str,
    ) -> HarnessResult:
        """带可靠性保障的 LLM 分析调用。

        流程：
            尝试调用 → 输出安全校验 → 通过则返回
                                  → 不通过则重试（最多 max_retries 次）
                                  → 全部失败则降级返回兜底报告

        参数：
            query_text     : 现场事件文本
            search_results : 向量检索结果列表
            rule_risk      : 规则系统判定的风险等级（只读）
            rule_action    : 规则系统判定的控制动作（只读）

        返回：
            HarnessResult（永远返回，不抛异常，保证主链路不中断）
        """
        started = time.perf_counter()

        # rule_risk / rule_action 只读：仅做空值兜底，不做任何改写
        risk_text = str(rule_risk or "").strip() or "未提供"
        action_text = str(rule_action or "").strip() or "未提供"

        last_report = ""
        last_validation = ValidationResult(passed=False)
        last_exception: Optional[str] = None      # 最后一次接口异常（超时/网络等）
        last_reject: Optional[str] = None         # 最后一次输出校验失败原因
        attempts = 0

        for attempt in range(1, self.max_retries + 2):     # 首次 + 重试
            attempts = attempt
            last_exception = None                          # 每次尝试前重置

            try:
                raw = self.agent.analyze(
                    query_text=query_text,
                    search_results=search_results,
                    rule_risk=risk_text,
                    rule_action=action_text,
                )
            except Exception as exc:  # noqa: BLE001  兜底：任何异常都不外抛
                last_exception = f"{type(exc).__name__}: {exc}"
                raw = ""

            # ---- 关键：识别 agent 内部已降级 ----
            # agv_agent.analyze() 会自行捕获异常并返回三段式兜底文案。
            # 该文案结构完整、无违规词，如果直接送校验会被判定为 success，
            # 从而把「LLM 失败」伪装成「分析成功」。这里显式识别并转为失败处理。
            if not last_exception and getattr(self.agent, "last_fallback", False):
                reason = getattr(self.agent, "last_fallback_reason", None) or "未知原因"
                last_exception = f"agent 内部兜底：{reason}"
                raw = ""      # 丢弃 agent 的兜底文案，改由本模块统一兜底

            last_report = str(raw or "")

            # ---- 输出安全校验（含篇幅观测）----
            validation = validate_report(last_report, max_chars=self.max_report_chars)
            last_validation = validation

            if validation.passed:
                result = HarnessResult(
                    report=last_report,
                    status="success",
                    attempts=attempts,
                    cost_ms=(time.perf_counter() - started) * 1000.0,
                    validation=validation,
                    report_chars=validation.report_chars,
                )
                self._capture_usage(result)
                self.stats.record(result)
                return result

            # 未通过：记录原因，准备重试
            if last_exception:
                # 接口本身失败，报告为空 —— 原因以异常为准，更有排查价值
                last_reject = f"接口异常：{last_exception}"
            else:
                last_reject = f"输出校验不通过：{validation.summary}"

            # 已是最后一次，不再等待
            if attempt <= self.max_retries:
                self._sleep_before_retry(attempt)

        # ---- 重试耗尽：降级兜底 ----
        fallback_report = _build_fallback_report(last_report, risk_text, action_text)
        result = HarnessResult(
            report=fallback_report,
            status="degraded",
            attempts=attempts,
            cost_ms=(time.perf_counter() - started) * 1000.0,
            validation=last_validation,
            report_chars=len(fallback_report.strip()),
            error=last_reject or "未知原因",
        )
        self._capture_usage(result)
        self.stats.record(result)
        return result

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _sleep_before_retry(self, attempt: int) -> None:
        """指数退避等待：1s → 2s → 4s，封顶 max_delay。"""
        if self.retry_base_delay <= 0:
            return
        delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.max_delay)
        time.sleep(delay)

    def _capture_usage(self, result: HarnessResult) -> None:
        """尝试从底层 agent 抓取 token 用量（若其暴露了最近一次调用信息）。

        说明：agv_agent 当前版本未返回 usage；这里做兼容式读取，
             取不到就保持 0，不报错。后续若 agent 暴露 usage 可自动生效。
        """
        usage = getattr(self.agent, "last_usage", None)
        if isinstance(usage, dict):
            result.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            result.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            result.total_tokens = int(
                usage.get("total_tokens", 0)
                or (result.prompt_tokens + result.completion_tokens)
            )

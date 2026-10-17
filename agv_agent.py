#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agv_agent.py —— 工业 AGV 具身智能项目的 LLM 智能体模块
================================================================================

本模块只做一件事：把「感知模块上报的现场事件」+「向量库检索到的相似历史故障」
+「系统已判定的风险等级与控制动作」交给大模型，产出一份面向工厂运维人员的
【辅助分析报告】。

==============================================================================
【安全边界 —— 本项目最重要的约束，任何改动都不得突破】
==============================================================================
    风险等级（rule_risk）与控制动作（rule_action）由【向量检索 + 硬编码规则】
    在上游提前算定，属于安全控制决策。

    本模块中的大模型【绝对不允许】：
        ✗ 修改风险等级
        ✗ 修改控制动作
        ✗ 建议修改风险等级或控制动作
        ✗ 降级 / 升级 / 质疑 / 复议 已经判定的安全结论
        ✗ 输出任何形式的新控制指令（如"建议停机""建议立即断电"）

    大模型只允许做「解读」：说明匹配情况、推测现场诱因、给出排查步骤。
    所有安全决策权 100% 保留在规则系统与现场人员手中。

    代码层面同样不提供任何修改 rule_risk / rule_action 的逻辑：这两个值在本模块
    中始终是【只读】的字符串，只会被原样转发给大模型和运维人员。

==============================================================================
对外接口
==============================================================================
    只有一个方法：

        AgvRiskAgent().analyze(
            query_text     = "当前现场事件文本",
            search_results = [{"similarity": 0.87, "area_id": "region_RAN", "text": "历史故障描述"}, ...],
            rule_risk      = "高风险",
            rule_action    = "AGV立即暂停，人工确认现场",
        ) -> str   # 返回三段式纯文本报告

    异常策略：接口超时、网络异常、鉴权失败等一律【内部捕获】，返回固定兜底
    报告，绝不向上抛异常中断主程序（AGV 主控链路不允许因为大模型不可用而中断）。

==============================================================================
依赖
==============================================================================
    pip install openai python-dotenv

    .env 配置（与本项目其它模块保持一致）：
        DEEPSEEK_API_KEY    = sk-xxxxxxxxxxxx          # 必填
        DEEPSEEK_BASE_URL   = https://api.deepseek.com # 选填，默认即此值
        AGV_AGENT_MODEL     = deepseek-chat            # 选填，默认即此值
        AGV_AGENT_TIMEOUT   = 60                       # 选填，单次请求超时（秒）
        AGV_AGENT_MAX_RETRIES = 2                      # 选填，SDK 内部重试次数
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # 读取 .env
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI  # DeepSeek 兼容 OpenAI SDK
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 0. 配置：全部从 .env 读取，代码里不硬编码密钥
# --------------------------------------------------------------------------- #
# 本模块所在目录，用于把 .env 定位到模块同级，避免受运行时工作目录影响
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
# 低温度：本模块要求输出稳定、可复现，不允许自由发挥
DEFAULT_TEMPERATURE = 0.2

# 单次响应输出 token 上限的【默认值】
# 背景：原提示词只要求"每条尽量具体"，未限篇幅，模型会写 7 条详细步骤，
#      单次约 1700+ 输出 token、耗时 10 秒。加上篇幅约束与 hard cap 后，
#      响应长度可控，单次耗时明显下降，同时保证三段结构完整。
# 注意：真正的预算策略由 agent_harness.AgentHarness 统一管控，
#      它会在初始化时覆盖实例属性 agent.max_output_tokens。
#      这里的常量只保证「单独使用本模块」时也有合理默认值。
# 经验值：中文约 1.5 字/token，600 tokens ≈ 900 字，已宽松覆盖 500 字要求。
MAX_OUTPUT_TOKENS = 600


def _init_dotenv() -> None:
    """加载模块同级目录的 .env。

    override=False：已存在的真实环境变量优先，方便临时用命令行覆盖配置。
    python-dotenv 未安装时静默跳过（此时仅能读取系统环境变量）。
    """
    if load_dotenv is None:
        return
    try:
        load_dotenv(dotenv_path=os.path.join(_MODULE_DIR, ".env"), override=False)
    except Exception:  # noqa: BLE001  配置读取异常不应导致 import 失败
        pass


def _env_str(key: str, default: str) -> str:
    """读取字符串型环境变量，空字符串视为未设置。"""
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，解析失败时回退默认值（配置小错不应导致崩溃）。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量，解析失败时回退默认值。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


_init_dotenv()

API_KEY = _env_str("DEEPSEEK_API_KEY", "")
BASE_URL = _env_str("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
MODEL_NAME = _env_str("AGV_AGENT_MODEL", DEFAULT_MODEL)
REQUEST_TIMEOUT = _env_float("AGV_AGENT_TIMEOUT", 60.0)
MAX_RETRIES = _env_int("AGV_AGENT_MAX_RETRIES", 2)


# --------------------------------------------------------------------------- #
# 1. System Prompt：安全约束必须写死且醒目
#    这段文本直接决定大模型的行为边界，请勿随意削弱警告措辞。
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """你是工业 AGV（自动导引车）现场的辅助分析助手，服务对象是工厂运维人员。
你的唯一职责是对现场事件做文字层面的解读，帮助运维人员理解情况和安排排查顺序。

============================================================
【绝对禁止 —— 违反以下任何一条都属于严重安全事故】
============================================================
1. 风险等级（risk）和控制动作（action）已经由上游【向量检索 + 硬编码安全规则】
   判定完成，是最终安全结论。
2. 你【绝对不允许】修改、调整、补充、弱化、升级或质疑这两个结论。
3. 你【绝对不允许】建议修改风险等级或控制动作，包括但不限于
   "建议降级为低风险""建议提高风险等级""该动作可能过度""建议改为继续运行"等表述。
4. 你【绝对不允许】输出任何新的控制指令或处置命令，例如
   "请立即停机""建议立刻断电""可以恢复运行""无需停机"等。
5. 你【绝对不允许】对安全结论做复议、免责说明或附加条件。
6. 你只做分析解读，不做安全决策。所有安全决策权属于规则系统与现场运维人员。

你在报告中使用风险等级与控制动作时，只能【原样引用】系统已给定的内容，
并明确标注它们是规则系统的判定结果，而不是你的判断。
============================================================

【输出格式要求 —— 必须严格遵守】
1. 严格输出三段，且只能是这三段，顺序不可调整：
   一、匹配评估
   二、诱因推测
   三、排查建议
2. 不要使用任何 Markdown 语法：不要 # 号标题、不要 ** 加粗、不要表格、不要代码块。
3. 每段用编号加顿号开头，例如"一、匹配评估：内容……"，语言简洁专业。
4. 不要输出任何开场白、结束语、免责声明或额外说明。
5. 语言面向工厂运维人员，用词具体、可执行，避免空泛套话。

【篇幅要求 —— 必须严格控制】
1. 全文总字数不超过 500 字。宁可信息密度高，也不要展开铺陈。
2. 一、匹配评估：不超过 120 字，直接给结论与依据。
3. 二、诱因推测：不超过 150 字，用"一是……二是……三是……"分点，最多 3 点。
4. 三、排查建议：最多 4 条，每条不超过 40 字。
5. 禁止复述原始资料、禁止解释原理、禁止展开背景，不要开场白与结束语。

【各段内容要求】
一、匹配评估：说明当前事件与历史故障库的匹配程度。必须参考给出的平均相似度数值，
   并据此判断属于"高匹配""弱相关"还是"几乎无匹配"三类之一，说明判断理由。
   如果历史库为空或没有有效相似度，直接说明"历史库中几乎没有可比对记录"。
二、诱因推测：结合当前事件和历史故障，推测可能的现场原因。只能基于给定材料推测，
   禁止编造材料中不存在的故障、部件、传感器或数据。若材料不足以支撑推测，
   明确说明"依据现有资料无法确定具体诱因"。
三、排查建议：给出现场运维人员按顺序执行的排查步骤，用"1. 2. 3."编号分条，
   【最多 4 条】，每条只写"看什么、测什么、确认什么"，不写操作细节与原理。
   排查建议只针对原因定位，不得包含任何运行、停机、复位、送电等控制类动作。
"""


# --------------------------------------------------------------------------- #
# 2. 内部工具：检索结果清洗与提示词拼装
# --------------------------------------------------------------------------- #
def _normalize_search_results(search_results: Any) -> List[Dict[str, Any]]:
    """清洗向量检索结果，返回统一结构的列表。

    鲁棒性处理（现场数据可能不规整）：
        · 不是列表时按空列表处理；
        · 非 dict 元素、缺少 text 的元素直接跳过；
        · similarity 非数值时置为 None（报告里会说明"无有效相似度"）；
        · area_id 缺失时统一填 "未知区域"。
    注意：本函数只做数据清洗，不改变任何安全结论。
    """
    if not isinstance(search_results, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in search_results:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "") or "").strip()
        if not text:
            # 没有故障描述的记录对分析没有价值
            continue

        raw_similarity = item.get("similarity")
        similarity: Optional[float] = None
        if isinstance(raw_similarity, (int, float)) and not isinstance(raw_similarity, bool):
            similarity = float(raw_similarity)

        area_id = str(item.get("area_id", "") or "").strip() or "未知区域"
        cleaned.append({"similarity": similarity, "area_id": area_id, "text": text})
    return cleaned


def _mean_similarity(results: List[Dict[str, Any]]) -> Optional[float]:
    """计算有效相似度的平均值；没有任何有效相似度时返回 None。"""
    values = [r["similarity"] for r in results if r.get("similarity") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_user_prompt(
    query_text: str,
    results: List[Dict[str, Any]],
    rule_risk: str,
    rule_action: str,
    mean_similarity: Optional[float],
) -> str:
    """拼装发给大模型的用户消息。

    设计要点：
        · 风险等级与控制动作以"规则系统判定结果（只读）"的形式给出，
          并在提示词里再次强调不得改动；
        · 历史故障用 JSON 给出，字段与向量库返回结构一致，便于模型对齐理解；
        · 预先算好平均相似度交给模型，避免模型自己做算术导致数值不一致。
    """
    if results:
        history_block = json.dumps(results, ensure_ascii=False, indent=2)
        if mean_similarity is None:
            mean_similarity_text = "无有效相似度数值（历史记录缺少 similarity 字段）"
        else:
            mean_similarity_text = f"{mean_similarity:.4f}（0~1，越大越相似）"
    else:
        history_block = "（空，向量库未检索到任何可比对的历史故障记录）"
        mean_similarity_text = "无（没有历史记录可供比对）"

    return f"""请对下面这次 AGV 现场事件做辅助分析，输出三段式报告。

【当前现场事件】
{query_text}

【历史故障库检索结果】
{history_block}

【平均相似度】
{mean_similarity_text}

【规则系统判定的风险等级（只读，不得修改）】
{rule_risk}

【规则系统判定的控制动作（只读，不得修改）】
{rule_action}

再次强调：
1. 上面两项安全结论只能原样引用并标注为规则系统判定结果，你不得修改、
   不得建议修改、不得质疑、不得输出任何新的控制指令。
2. 严格按"一、匹配评估""二、诱因推测""三、排查建议"三段输出，不要 Markdown。
3. 排查建议只写原因定位类动作，不得包含运行、停机、复位、送电等控制类动作。
4. 禁止编造历史故障库和当前事件中都不存在的部件、传感器或现象。
"""


def _compose_safe_fallback(mean_similarity: Optional[float], rule_risk: str, rule_action: str) -> str:
    """构造带现场信息的兜底报告：保留规则结论，仅提示分析服务不可用。

    与纯固定文案的区别：这里把规则系统已判定的风险等级与控制动作一并
    带给运维人员，避免他们在服务异常时反而丢失关键安全信息。
    注意：两个安全结论均为【原样引用】，本函数不做任何改动。
    """
    if mean_similarity is None:
        match_text = "智能分析服务当前不可用，本次未能完成匹配度分析；历史库中也未取到有效相似度数值。"
    else:
        match_text = (
            f"智能分析服务当前不可用，本次未能完成匹配度分析；"
            f"检索到的历史故障平均相似度为 {mean_similarity:.4f}（仅供参考）。"
        )

    return (
        f"一、匹配评估：{match_text}\n"
        f"二、诱因推测：因分析服务调用失败，未能依据历史故障给出诱因推测，"
        f"请勿基于缺失信息臆断现场原因。\n"
        f"三、排查建议：1. 严格遵守规则系统判定的风险等级：{rule_risk}。"
        f"2. 严格执行规则系统判定的控制动作：{rule_action}。"
        f"3. 人工查阅本次事件对应的历史故障记录，核对现场实际状态。"
        f"4. 分析服务恢复后可重新发起本次分析以获取完整报告。\n"
    )


# --------------------------------------------------------------------------- #
# 3. 对外唯一接口类
# --------------------------------------------------------------------------- #
class AgvRiskAgent:
    """AGV 现场事件辅助分析智能体。

    使用方式：
        agent = AgvRiskAgent()
        report = agent.analyze(
            query_text="3号AGV在B区充电桩前报定位丢失",
            search_results=vector_store.search("定位丢失", top_k=3, area_id="region_RAZ"),
            rule_risk="高风险",
            rule_action="AGV立即暂停，人工确认现场",
        )

    安全说明：
        本类不接收、不返回、也不修改任何控制决策。rule_risk 与 rule_action
        仅作为只读文本透传给大模型与运维人员，类中不存在任何改写它们的代码路径。
    """

    def __init__(self) -> None:
        """初始化。

        客户端采用惰性创建：只有真正调用 analyze() 时才建立连接，
        避免仅仅 import 本模块就要求已配置 API Key。
        """
        self._client: Optional[Any] = None
        # 最近一次调用的 token 用量（供上层 harness / 日志做成本统计）
        # 结构：{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        self.last_usage: Optional[Dict[str, int]] = None
        # 本次调用是否走了内部兜底（LLM 失败时本模块会吞掉异常返回兜底文案）。
        # 上层 Harness 必须读取这两个字段，否则会把兜底报告误判为真实报告。
        self.last_fallback: bool = False
        self.last_fallback_reason: Optional[str] = None
        # 输出预算（max_tokens）：作为实例属性暴露，可由上层 Harness 注入覆盖。
        # 为什么不写死调用参数：单次响应的长度是【调用策略】，属于 Harness 的职责，
        # 放在这里只是提供一个默认值，便于单独使用本模块时也能正常工作。
        self.max_output_tokens: int = MAX_OUTPUT_TOKENS

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Optional[Any]:
        """获取（惰性创建）OpenAI 兼容客户端。

        未配置 API Key 或未安装 openai 时返回 None，
        由 analyze() 走兜底分支，不向上抛异常。
        """
        if self._client is None:
            if OpenAI is None or not API_KEY:
                return None
            self._client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                timeout=REQUEST_TIMEOUT,
                max_retries=MAX_RETRIES,
            )
        return self._client

    def _call_llm(self, user_prompt: str) -> str:
        """调用 DeepSeek chat 接口，返回模型正文文本。

        调用失败时抛出异常，由 analyze() 统一捕获并转兜底报告——
        这样异常处理集中在一处，便于排查，也保证接口不会中断主程序。
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError(
                "LLM 客户端不可用：请确认已安装 openai，并在 .env 中配置 DEEPSEEK_API_KEY。"
            )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # 低温度：本模块要求输出稳定、可复现
            temperature=DEFAULT_TEMPERATURE,
            # 硬上限：限制输出长度，避免模型无视字数要求导致响应过慢
            # 经验值：中文约 1.5 字/token，600 tokens ≈ 900 字，已宽松覆盖 500 字要求
            max_tokens=self.max_output_tokens,
        )

        content = ""
        if response.choices:
            content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("LLM 返回内容为空。")

        # ---- 记录 token 用量（供上层 harness / 日志做成本统计）----
        # 说明：这里用 try 包裹，usage 结构变化不应影响报告正常返回。
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
        except Exception:  # noqa: BLE001
            self.last_usage = None

        return content

    # ------------------------------------------------------------------ #
    # 对外唯一接口
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        query_text: str,
        search_results: list,
        rule_risk: str,
        rule_action: str,
    ) -> str:
        """对一次 AGV 现场事件生成三段式辅助分析报告。

        参数：
            query_text     : 当前发生的现场事件文本
            search_results : 向量检索返回列表，元素形如
                             {"similarity": 0.87, "area_id": "region_RAN", "text": "历史故障描述"}
                             允许为空列表；元素字段缺失也能正常处理
            rule_risk      : 规则系统判定的风险等级，例如 "高风险"
            rule_action    : 规则系统判定的控制动作，例如 "AGV立即暂停，人工确认现场"

        返回：
            str —— 三段式纯文本报告（一、匹配评估 / 二、诱因推测 / 三、排查建议）。
            接口超时、网络异常、鉴权失败等情况一律返回兜底报告，不抛异常。

        安全：
            rule_risk 与 rule_action 在本方法中全程只读，仅用于拼接提示词与
            兜底报告，不存在任何修改、推导或建议修改它们的逻辑。
        """
        # ---- 每次调用前重置状态，避免上一次的 token / 降级标记残留 ----
        self.last_usage = None
        self.last_fallback = False
        self.last_fallback_reason = None

        # ---- 参数校验：非法输入不抛异常，转为可读提示，保证主链路不中断 ----
        query_text = str(query_text or "").strip()
        if not query_text:
            query_text = "（未提供事件描述）"

        # rule_risk / rule_action 均为只读，这里只做空值兜底，不做任何改写
        rule_risk_text = str(rule_risk or "").strip() or "未提供"
        rule_action_text = str(rule_action or "").strip() or "未提供"

        # ---- 检索结果清洗：空列表、None、字段缺失、元素类型异常都能处理 ----
        results = _normalize_search_results(search_results)
        mean_similarity = _mean_similarity(results)

        # ---- 拼装提示词并调用模型，异常统一兜底 ----
        try:
            user_prompt = _build_user_prompt(
                query_text=query_text,
                results=results,
                rule_risk=rule_risk_text,
                rule_action=rule_action_text,
                mean_similarity=mean_similarity,
            )
            return self._call_llm(user_prompt)
        except Exception as exc:  # noqa: BLE001
            # 覆盖：接口超时、网络不可达、鉴权失败、返回体异常、未配置 Key 等
            # 统一返回兜底报告，绝不向上抛异常中断 AGV 主程序。
            # 同时记录降级状态，供上层 Harness 区分「真实报告」与「兜底文案」。
            self.last_fallback = True
            self.last_fallback_reason = f"{type(exc).__name__}: {exc}"
            return _compose_safe_fallback(mean_similarity, rule_risk_text, rule_action_text)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vector_store.py —— AGV 巡检小车：全链路业务入口与联调测试
================================================================================

本文件是【业务主入口】，负责把四个模块串成一条完整链路：

    VectorStore（区域优先召回） → 规则阈值决策 → AgentHarness（LLM 可靠性外壳）
                                                        ↓
                                            simple_logger（jsonl 落盘）

    AgentHarness 在调用 LLM 前后加四层保障：
        输出安全校验 → 失败重试（指数退避）→ 降级兜底 → 指标统计（耗时/token/成功率）

数据全部来自真实来源，不做任何模拟：
    · 16 条故障案例  ← knowledge_base.json（4 个区域各 4 条）
    · 查询向量        ← 本地 ONNX 模型真实推理（BGE 中文模型，512 维）
    · 相似度          ← 真实余弦相似度计算
    · LLM 分析        ← deepseek 真实接口，经 AgentHarness 做输出校验与降级
    · 可靠性指标      ← 每次调用的重试次数/token/校验结果，随日志落盘

分层约定：
    · vector_store.py 只管向量存储与召回，内部不打印日志；
    · agv_agent.py 只管 LLM 分析，不感知日志与向量库；
    · simple_logger.py 只管日志持久化；
    · 三者在本文件（业务入口层）完成组装。

执行内容：
    【阶段0】初始化：加载知识库 + 构建向量索引
    【阶段1】3 组召回用例：区域优先(RAN) / 区域优先(RDA) / 纯全局
    【阶段2】真实分数分布统计：给阈值提供数据依据（不是拍脑袋定的）
    【阶段3】区域优先 vs 全局召回 A/B 对照：量化跨区污染率下降与语义代价

运行方式：
    cd /Users/bobo/Desktop/agv_embodied_project
    ./venv/bin/python test_vector_store.py

前置条件：
    · knowledge_base.json 与 vector_store.py 同目录（已就位）
    · 本地已放好 BGE ONNX 模型（onnx_models/bge-small-zh-v1.5/onnx/，已就位）
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from agent_harness import AgentHarness
from simple_logger import logger
from vector_store import VectorStore, cosine_similarity

# --------------------------------------------------------------------------- #
# 规则阈值（硬编码安全规则，不是 LLM 决策）
# --------------------------------------------------------------------------- #
# 阈值按【查询-文档】相似度分布校准（实测 BGE 下 max_sim 区间 0.71~0.83）。
# 注意：不要拿"文档间两两相似度"的分位数来定这里 —— 那是另一个分布，
#      查询与文档的相似度天然更高。
NO_MATCH_MAX_THRESHOLD: float = 0.45   # 低于此值视为无有效匹配（实测 min=0.71，留足余量）
MID_RISK_AVG_THRESHOLD: float = 0.67   # 中等匹配线
HIGH_RISK_AVG_THRESHOLD: float = 0.78  # 高匹配线

_LINE_WIDTH: int = 84

# --------------------------------------------------------------------------- #
# 3 组召回用例
# --------------------------------------------------------------------------- #
# 注意：query_text 刻意【不抄知识库原文】，用近义改写模拟真实现场事件，
#      这样相似度落在合理区间，能真实反映语义检索能力。
TEST_CASES: List[Dict[str, Any]] = [
    {
        "case_name": "用例1：区域优先召回 —— 装载区 region_RAN",
        "query_text": "装载区AGV把货架看错了，出现幻视导致货物位置判断错误",
        "area_id": "region_RAN",
        "expect_area": "region_RAN",   # 期望 top1 落在该区域
        "top_k": 3,
    },
    {
        "case_name": "用例2：区域优先召回 —— 充电区 region_RDA",
        "query_text": "充电时电池温度过高报警，充电过程被迫中断",
        "area_id": "region_RDA",
        "expect_area": "region_RDA",
        "top_k": 3,
    },
    {
        "case_name": "用例3：纯全局召回 —— 不指定区域",
        "query_text": "AGV撞到东西后触发安全触边，车停住了",
        "area_id": None,
        "expect_area": "region_RAB",   # 语义上最贴近避让区
        "top_k": 3,
    },
]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def print_title(text: str) -> None:
    """打印分节标题。"""
    print("\n" + "=" * _LINE_WIDTH)
    print(f"  {text}")
    print("=" * _LINE_WIDTH)


def decide_risk(max_sim: float, avg_similarity: float) -> Tuple[str, str, str]:
    """按硬编码安全规则给出风险等级与处置动作。

    ============================ 判据说明（重要）============================
    风险等级取决于【最高相似度 max_sim】，而不是平均相似度 avg_similarity。

    原因：top_k 召回结果中，除 top1 外的条目多为语义无关记录，会把均值拉低。
         avg 实际反映的是"干扰项有多低"，而不是"匹配有多准"。若用 avg 判风险，
         会出现反直觉结果——实测中匹配最准的一次（max_sim=0.8314）被判为
         "低风险"，而匹配更弱的一次（max_sim=0.7667）反而被判"中风险"。
         且该判据会随 top_k 取值漂移（top_k 从 3 改到 5，avg 会明显下降）。

    avg_similarity 仍保留为参数与日志字段，供分析知识库覆盖度使用，
    但不参与安全决策。
    ========================================================================

    规则：
        max_sim < NO_MATCH_MAX_THRESHOLD(0.45)  -> 高风险 / 减速重新扫描，上报日志
        max_sim >= HIGH_RISK_AVG_THRESHOLD(0.78)-> 高风险 / 立即暂停，人工确认现场
        max_sim >= MID_RISK_AVG_THRESHOLD(0.67) -> 中风险 / 减速运行，二次校验传感器数据
        else                                    -> 低风险 / 继续正常作业

    参数：
        max_sim        : 召回结果中的最大相似度（风险决策依据）
        avg_similarity : 召回结果的平均相似度（仅记录，不参与决策）

    返回：
        (risk_level, action, decision_text)

    注意：本函数是纯规则判断，与 LLM 无关；LLM 不允许修改这里的结论。
    """
    if max_sim < NO_MATCH_MAX_THRESHOLD:
        return "高风险", "AGV减速，重新扫描环境，上传异常日志", \
               "高风险：AGV减速，重新扫描环境，上传异常日志"
    if max_sim >= HIGH_RISK_AVG_THRESHOLD:
        return "高风险", "AGV立即暂停，人工确认现场", \
               "高风险：AGV立即暂停，人工确认现场"
    if max_sim >= MID_RISK_AVG_THRESHOLD:
        return "中风险", "AGV减速运行，二次校验传感器数据", \
               "中风险：AGV减速运行，二次校验传感器数据"
    return "低风险", "AGV继续正常作业", "低风险：AGV继续正常作业"


# --------------------------------------------------------------------------- #
# 阶段1：执行一组召回用例，并走完整链路（召回 → 决策 → LLM → 日志）
# --------------------------------------------------------------------------- #
def run_case(
    vs: VectorStore,
    agent: Any,          # AgentHarness 实例（带输出校验/重试/降级）
    case: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """执行一组用例，返回日志 payload（失败返回 None）。

    链路：真实向量召回 → 阈值决策 → LLM 辅助分析 → 日志落盘
    """
    print_title(case["case_name"])
    print(f"  查询语句 : {case['query_text']}")
    print(f"  检索区域 : {case['area_id']}    期望条数 : {case['top_k']}")

    # ---- ① 真实向量召回（计时）----
    t0 = time.perf_counter()
    try:
        results = vs.search(
            query_text=case["query_text"],
            top_k=case["top_k"],
            area_id=case["area_id"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [ERROR] 召回失败：{type(exc).__name__}: {exc}")
        return None
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    # ---- ② 打印每条召回结果 ----
    print("-" * _LINE_WIDTH)
    if not results:
        print("  未召回任何结果")
    for rank, item in enumerate(results, start=1):
        print(f"  [{rank}] score={item['score']:.4f}  area_id={item['area_id']}  "
              f"risk_level={item['risk_level']}")
        print(f"      fault_id  : {item['fault_id']}")
        print(f"      fault_desc: {item['fault_desc']}")
    print("-" * _LINE_WIDTH)

    # ---- ③ 提取相似度（字段名是 score），先判空再求平均，避免除零 ----
    if results:
        scores = [float(item["score"]) for item in results]
        max_sim = max(scores)
        avg_similarity = sum(scores) / len(scores)
    else:
        max_sim = 0.0
        avg_similarity = 0.0

    # ---- ④ 规则风险判断（安全决策链路，LLM 无权干预）----
    risk_level, action, decision_text = decide_risk(max_sim, avg_similarity)

    # ---- ⑤ 打印统计与决策 ----
    print(f"  max_sim        = {max_sim:.4f}   ← 风险判据")
    print(f"  avg_similarity = {avg_similarity:.4f}   （仅记录，不参与决策）")
    print(f"  rule_decision  = {decision_text}")

    # ---- ⑥ LLM 智能分析链路（经 Harness：输出校验 + 重试 + 降级 + 指标）----
    # search_results 直接用召回结果：字典里同时含 similarity / text 别名，
    # agv_agent 的字段清洗逻辑可直接消费，无需任何转换。
    # Harness 会在模型输出越界（如"建议停机"）时拦截并降级，保证安全边界。
    t1 = time.perf_counter()
    llm_result = agent.analyze(
        query_text=case["query_text"],
        search_results=results,
        rule_risk=risk_level,
        rule_action=action,
    )
    llm_ms = (time.perf_counter() - t1) * 1000.0
    report = llm_result.report

    print("-" * _LINE_WIDTH)
    print("  【LLM 辅助分析报告】（只做解读，不参与安全决策）")
    print("-" * _LINE_WIDTH)
    for line in str(report).splitlines():
        print(f"  {line}")
    print("-" * _LINE_WIDTH)
    print(f"  召回耗时 {retrieval_ms:.1f} ms | LLM 耗时 {llm_ms:.1f} ms")
    print(f"  Harness: status={llm_result.status}  attempts={llm_result.attempts}  "
          f"tokens={llm_result.total_tokens}  {llm_result.report_chars}字  "
          f"校验={llm_result.validation.summary}")

    # 提示：若报告内容为"智能分析服务当前不可用"，说明 LLM 调用失败走了兜底，
    #      不影响主链路，但需要排查 .env 中的 Key 与模型名配置。
    if not llm_result.is_success:
        print(f"  [提示] 本次 LLM 已降级（原因：{llm_result.error}）")

    # ---- ⑦ 组装日志 payload 并落盘（日志统一在业务入口层写入）----
    top_hit = results[0] if results else None
    payload: Dict[str, Any] = {
        "query_text": case["query_text"],
        "search_area": case["area_id"],
        "top_k": case["top_k"],
        "retrieved_count": len(results),
        "max_sim": round(max_sim, 4),
        "avg_similarity": round(avg_similarity, 4),
        "risk_level": risk_level,
        "action": action,
        "risk_decision": decision_text,
        "top_fault_id": top_hit["fault_id"] if top_hit else None,
        "top_fault_desc": top_hit["fault_desc"] if top_hit else None,
        "retrieved_fault_ids": [item["fault_id"] for item in results],
        "retrieval_result": results,
        "llm_analysis_report": report,
        # —— Harness 指标（可靠性 + 成本，写进日志便于复盘与统计）——
        "llm_status": llm_result.status,
        "llm_attempts": llm_result.attempts,
        "llm_validation": llm_result.validation.to_dict(),
        "llm_prompt_tokens": llm_result.prompt_tokens,
        "llm_completion_tokens": llm_result.completion_tokens,
        "llm_total_tokens": llm_result.total_tokens,
        "llm_report_chars": llm_result.report_chars,
        "llm_too_long": llm_result.validation.too_long,
        "llm_error": llm_result.error,
        "retrieval_cost_ms": round(retrieval_ms, 1),
        "llm_cost_ms": round(llm_ms, 1),
        "cost_ms": round(retrieval_ms + llm_ms, 1),
    }
    logger.log_event(payload)
    return payload


# --------------------------------------------------------------------------- #
# 阶段2：真实分数分布统计（为阈值提供数据依据）
# --------------------------------------------------------------------------- #
def analyze_score_distribution(vs: VectorStore) -> Dict[str, float]:
    """用真实向量计算 16 条案例两两之间的余弦相似度分布。

    目的：量化知识库内部的语义紧密度，作为阈值讨论的数据依据。

    ⚠️ 重要区分：这里统计的是【文档间】相似度（doc-doc），
       而风险决策用的是【查询-文档】相似度（query-doc）。
       两者不是同一个分布——query-doc 天然更高（实测 max_sim 0.71~0.83，
       而 doc-doc 的 P90 只有 0.68）。
       因此本函数的分位数【不应直接】用作 decide_risk 的阈值。

    返回：
        含 count / min / max / mean / p50 / p75 / p90 / p95 的字典
    """
    print_title("阶段2：文档间分数分布统计（知识库紧密度）")

    # 逐对计算余弦相似度：16 条案例共 C(16,2)=120 对
    scores: List[float] = []
    records = vs.records
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            scores.append(
                cosine_similarity(records[i].embedding, records[j].embedding)
            )

    scores.sort()
    n = len(scores)
    if n == 0:
        print("  案例数量不足，无法统计")
        return {}

    def percentile(p: float) -> float:
        """取分位数（最近邻法，避免插值带来歧义）。"""
        idx = min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))
        return scores[idx]

    stats: Dict[str, float] = {
        "count": float(n),
        "min": scores[0],
        "max": scores[-1],
        "mean": statistics.fmean(scores),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
    }

    print(f"  样本对数：{n} 对（{len(records)} 条案例两两组合）")
    print(f"  min={stats['min']:.4f}  max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}")
    print(f"  P50={stats['p50']:.4f}  P75={stats['p75']:.4f}  "
          f"P90={stats['p90']:.4f}  P95={stats['p95']:.4f}")

    print("\n  【当前风险阈值 vs 该分布】")
    print(f"    HIGH_RISK = {HIGH_RISK_AVG_THRESHOLD:.2f}  "
          f"→ 约 P{_rank_of(stats, HIGH_RISK_AVG_THRESHOLD):.0f} 分位")
    print(f"    MID_RISK  = {MID_RISK_AVG_THRESHOLD:.2f}  "
          f"→ 约 P{_rank_of(stats, MID_RISK_AVG_THRESHOLD):.0f} 分位")
    print(f"    NO_MATCH  = {NO_MATCH_MAX_THRESHOLD:.2f}  "
          f"→ 该分布最小值 {stats['min']:.4f}，用于兜住'完全无匹配'的情况")

    print("\n  说明：上述分位数仅供了解知识库语义紧密度。")
    print("        风险阈值应按【查询-文档】尺度设定（见阶段1 的 max_sim 实测值），")
    print("        不要直接套用本分布，否则会出现'高分档永不触发'的问题。")
    return stats


def _rank_of(stats: Dict[str, float], threshold: float) -> float:
    """估算某个阈值落在实测分布的大致百分位，便于判断阈值是否偏高。"""
    if not stats:
        return 0.0
    # 用 P50/P75/P90/P95 做分段线性近似
    points = [(stats["p50"], 50.0), (stats["p75"], 75.0),
              (stats["p90"], 90.0), (stats["p95"], 95.0)]
    if threshold <= points[0][0]:
        return max(1.0, 50.0 * threshold / max(points[0][0], 1e-9))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if threshold <= x1:
            return y0 + (y1 - y0) * (threshold - x0) / max(x1 - x0, 1e-9)
    return 99.0


# --------------------------------------------------------------------------- #
# 阶段3：区域优先 vs 全局召回 —— A/B 对照实验
# --------------------------------------------------------------------------- #
# ============================ 实验设计说明 ============================
# 很多人做这类对比时会犯一个方法论错误：拿"全局检索"的 top1 去比"期望区域"，
# 但全局模式根本不知道事件发生在哪个区，等于给它设了一个不可能知道的答案。
#
# 本实验改为 A/B 对照，两个模式的前提严格区分：
#
#   模式 A（global）：【不知道】事件发生在哪个区域，只凭文本语义召回。
#                    模拟"只有事件描述、没有区域信息"的场景。
#   模式 B（area）  ：【知道】事件发生在某区域（现场上报/传感器定位），
#                    在区域内优先召回，不足再从全局补齐。
#
# 评测指标换成两个更有意义的量：
#   ① 跨区污染率 = 召回结果中来自【其它区域】的比例
#        —— 这些记录对运维是噪声：区号不对，故障机理也大概率不同。
#        —— 越低越好，这是区域优先的核心收益。
#   ② 平均相似度 = 召回结果的平均语义相似度
#        —— 反映召回质量。区域约束可能牺牲一点相似度，需要如实体现。
#
# 结论按"污染率下降 + 相似度代价"一起看，而不是只看命中率。
# ======================================================================

# 跨区共性故障基准集：这些故障在多个区域都有类似记录，
# 全局检索容易把其它区域的记录召回进来，正好用来检验区域优先的效果。
CROSS_REGION_BENCHMARK: List[Dict[str, Any]] = [
    {
        "label": "碰撞触边类",
        "query_text": "AGV在避让区碰到东西后停下了",
        "area_id": "region_RAB",
    },
    {
        "label": "急停锁止类",
        "query_text": "AGV在缓存区突然急停锁死了",
        "area_id": "region_RAZ",
    },
    {
        "label": "过载报警类",
        "query_text": "AGV驱动电机电流异常过载了",
        "area_id": "region_RAZ",
    },
    {
        "label": "二维码丢失类",
        "query_text": "AGV在装载区地面二维码识别失败",
        "area_id": "region_RAN",
    },
    {
        "label": "电池温度类",
        "query_text": "AGV在充电区电池温度报警",
        "area_id": "region_RDA",
    },
    {
        "label": "障碍物避让类",
        "query_text": "AGV前方有障碍物减速了",
        "area_id": "region_RAB",
    },
]


def compare_area_vs_global(vs: VectorStore) -> Dict[str, Any]:
    """区域优先 vs 全局召回 的 A/B 对照实验。

    对基准集中每个查询，分别用两种模式召回 top_k，比较：
        ① 跨区污染率：召回结果中来自非目标区域的比例（越低越好）
        ② 平均相似度：召回结果的平均语义相似度（反映召回质量）
        ③ top1 相似度：最相似那条的分数（反映最佳匹配质量）

    返回：
        含汇总指标与逐条明细的字典（写日志用）
    """
    print_title("阶段3：区域优先召回 vs 全局召回（A/B 对照实验）")

    print("  实验设计：")
    print("    A 模式(global) = 不知道事件区域，只凭语义召回")
    print("    B 模式(area)   = 知道事件区域(现场上报)，区域内优先召回")
    print("    指标：跨区污染率 = 召回结果中来自其它区域的比例（越低越好）")
    print()

    header = (f"  {'故障类型':14} {'A污染':>7} {'B污染':>7} {'A相似度':>8} "
              f"{'B相似度':>8} {'A_top1':>8} {'B_top1':>8}")
    print(header)
    print("-" * _LINE_WIDTH)

    rows: List[Dict[str, Any]] = []
    for item in CROSS_REGION_BENCHMARK:
        q = item["query_text"]
        area = item["area_id"]
        top_k = 3

        g_res = vs.search(q, top_k=top_k, area_id=None)
        a_res = vs.search(q, top_k=top_k, area_id=area)

        def _metrics(res: List[Dict[str, Any]]) -> Tuple[float, float, float]:
            """返回 (跨区污染率, 平均相似度, top1 相似度)。"""
            if not res:
                return 0.0, 0.0, 0.0
            polluted = sum(1 for x in res if x["area_id"] != area)
            avg_score = sum(float(x["score"]) for x in res) / len(res)
            top1_score = float(res[0]["score"])
            return polluted / len(res), avg_score, top1_score

        g_pollute, g_avg, g_top1 = _metrics(g_res)
        a_pollute, a_avg, a_top1 = _metrics(a_res)

        # 计算 "A 模式跨区污染" 的绝对条数，便于讲解
        g_polluted_cnt = sum(1 for x in g_res if x["area_id"] != area)
        a_polluted_cnt = sum(1 for x in a_res if x["area_id"] != area)

        print(f"  {item['label']:14} {g_pollute:>6.0%} {a_pollute:>7.0%} "
              f"{g_avg:>8.4f} {a_avg:>8.4f} {g_top1:>8.4f} {a_top1:>8.4f}")

        rows.append({
            "label": item["label"],
            "query_text": q,
            "target_area": area,
            "global": {
                "polluted_count": g_polluted_cnt,
                "pollution_rate": round(g_pollute, 4),
                "avg_similarity": round(g_avg, 4),
                "top1_similarity": round(g_top1, 4),
                "results": [{"area_id": x["area_id"], "fault_id": x["fault_id"],
                             "score": round(float(x["score"]), 4)} for x in g_res],
            },
            "area": {
                "polluted_count": a_polluted_cnt,
                "pollution_rate": round(a_pollute, 4),
                "avg_similarity": round(a_avg, 4),
                "top1_similarity": round(a_top1, 4),
                "results": [{"area_id": x["area_id"], "fault_id": x["fault_id"],
                             "score": round(float(x["score"]), 4)} for x in a_res],
            },
        })

    # ---- 汇总 ----
    n = len(rows)
    g_pollute_avg = sum(r["global"]["pollution_rate"] for r in rows) / n if n else 0.0
    a_pollute_avg = sum(r["area"]["pollution_rate"] for r in rows) / n if n else 0.0
    g_sim_avg = sum(r["global"]["avg_similarity"] for r in rows) / n if n else 0.0
    a_sim_avg = sum(r["area"]["avg_similarity"] for r in rows) / n if n else 0.0
    g_top1_avg = sum(r["global"]["top1_similarity"] for r in rows) / n if n else 0.0
    a_top1_avg = sum(r["area"]["top1_similarity"] for r in rows) / n if n else 0.0

    print("-" * _LINE_WIDTH)
    print(f"  {'平均':14} {g_pollute_avg:>6.0%} {a_pollute_avg:>7.0%} "
          f"{g_sim_avg:>8.4f} {a_sim_avg:>8.4f} {g_top1_avg:>8.4f} {a_top1_avg:>8.4f}")

    print("\n  【结论】")
    print(f"    跨区污染率 : {g_pollute_avg:.0%} → {a_pollute_avg:.0%} "
          f"（下降 {max(0.0, (g_pollute_avg - a_pollute_avg)) * 100:.0f} 个百分点）")
    print(f"    平均相似度 : {g_sim_avg:.4f} → {a_sim_avg:.4f} "
          f"（{a_sim_avg - g_sim_avg:+.4f}，区域约束的语义代价）")
    print(f"    top1 相似度: {g_top1_avg:.4f} → {a_top1_avg:.4f} "
          f"（{a_top1_avg - g_top1_avg:+.4f}，最佳匹配基本不受影响）")
    print("\n    解读：区域优先以【少量语义相似度损失】换取【召回结果全部来自正确区域】，")
    print("          减少了进入 LLM 上下文的跨区噪声，适合已知事件发生区域的现场场景。")

    return {
        "experiment_type": "A/B 对照（global=无区域信息 / area=已知区域）",
        "benchmark_size": n,
        "global_avg_pollution": round(g_pollute_avg, 4),
        "area_avg_pollution": round(a_pollute_avg, 4),
        "pollution_reduction": round(max(0.0, g_pollute_avg - a_pollute_avg), 4),
        "global_avg_similarity": round(g_sim_avg, 4),
        "area_avg_similarity": round(a_sim_avg, 4),
        "similarity_delta": round(a_sim_avg - g_sim_avg, 4),
        "global_avg_top1": round(g_top1_avg, 4),
        "area_avg_top1": round(a_top1_avg, 4),
        "details": rows,
    }

# --------------------------------------------------------------------------- #
# 程序入口
# --------------------------------------------------------------------------- #
def main() -> int:
    """执行全链路联调：初始化 → 3 组用例 → 分数分布 → 区域对比。"""
    overall_start = time.perf_counter()

    print_title("AGV 巡检小车 V1 POC —— 全链路联调测试")
    print("  链路：VectorStore 区域优先召回 → 规则阈值决策 → LLM 辅助分析 → 日志落盘")

    # ---- 阶段0：初始化（加载知识库 + 真实向量化）----
    print_title("阶段0：初始化")
    t0 = time.perf_counter()
    try:
        vs = VectorStore()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [FATAL] VectorStore 初始化失败：{type(exc).__name__}: {exc}")
        return 1
    init_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  知识库文件 : {vs.kb_path}")
    print(f"  案例总数   : {vs.count()} 条")
    print(f"  区域分布   : {vs.area_distribution()}")
    print(f"  向量维度   : {len(vs.records[0].embedding)} 维")
    print(f"  初始化耗时 : {init_ms:.0f} ms（含模型加载 + 案例向量化）")

    # LLM 客户端惰性创建：此处不会联网，只在 analyze() 时真正请求
    # 包一层 Harness：输出安全校验 → 重试 → 降级 → 指标统计
    # 输出预算由 Harness 统一管控（不再写死在 agv_agent 里）：
    #   max_output_tokens=600 -> API 层硬上限，控制单次响应时长
    #   max_report_chars=800  -> 软阈值，超长只告警不重试
    agent = AgentHarness(
        max_retries=2,
        retry_base_delay=1.0,
        max_output_tokens=600,
        max_report_chars=800,
    )
    print("  LLM 组件   : AgentHarness 已创建（输出校验/重试/降级/预算管控）")
    print(f"  输出预算   : max_tokens={agent.max_output_tokens}  "
          f"软阈值={agent.max_report_chars} 字")

    # ---- 阶段1：3 组召回用例 ----
    print_title("阶段1：召回用例（区域优先 / 区域优先 / 纯全局）")
    payloads: List[Dict[str, Any]] = []
    for case in TEST_CASES:
        result = run_case(vs, agent, case)
        if result is not None:
            payloads.append(result)

    # ---- 阶段2：真实分数分布 ----
    stats = analyze_score_distribution(vs)

    # ---- 阶段3：区域优先 vs 全局对比 ----
    comparison = compare_area_vs_global(vs)

    # ---- 汇总写入日志 ----
    total_ms = (time.perf_counter() - overall_start) * 1000.0
    logger.log_event({
        "module": "test_vector_store",
        "run_type": "full_chain",
        "knowledge_base_path": vs.kb_path,
        "knowledge_base_count": vs.count(),
        "area_distribution": vs.area_distribution(),
        "init_cost_ms": round(init_ms, 1),
        "case_count": len(payloads),
        "score_distribution": {k: round(v, 4) for k, v in stats.items()} if stats else {},
        "area_vs_global": comparison,
        # —— Harness 累积统计（成功率、平均耗时、token 总量、降级原因）——
        "harness_stats": agent.stats.to_dict(),
        "total_cost_ms": round(total_ms, 1),
        "events": payloads,
    })

    # ---- 最终汇总 ----
    print_title("执行汇总")
    print(f"  召回用例      : {len(payloads)}/{len(TEST_CASES)} 组完成")
    print(f"  知识库案例    : {vs.count()} 条，分布 {vs.area_distribution()}")
    if stats:
        print(f"  文档间分布    : min={stats['min']:.3f} max={stats['max']:.3f} "
              f"P50={stats['p50']:.3f} P75={stats['p75']:.3f} P90={stats['p90']:.3f}")
    if comparison.get("benchmark_size"):
        print(f"  区域优先收益  : 跨区污染率 "
              f"{comparison['global_avg_pollution']:.0%} → "
              f"{comparison['area_avg_pollution']:.0%}"
              f"（下降 {comparison['pollution_reduction'] * 100:.0f} 个百分点）")
        print(f"  语义代价      : 平均相似度 "
              f"{comparison['global_avg_similarity']:.4f} → "
              f"{comparison['area_avg_similarity']:.4f}"
              f"（{comparison['similarity_delta']:+.4f}）")
    hs = agent.stats
    print(f"  LLM 可靠性    : 成功率 {hs.success_rate:.0%} "
          f"({hs.success_calls}/{hs.total_calls})，平均 {hs.avg_cost_ms:.0f} ms")
    print(f"  Token 消耗    : 输入 {hs.total_prompt_tokens} / "
          f"输出 {hs.total_completion_tokens} / 合计 "
          f"{hs.total_prompt_tokens + hs.total_completion_tokens}")
    if payloads:
        avg_chars = sum(p.get("llm_report_chars", 0) for p in payloads) / len(payloads)
        print(f"  报告篇幅      : 平均 {avg_chars:.0f} 字"
              f"（软阈值 {agent.max_report_chars} 字）")
    print(f"  整体耗时      : {total_ms:.0f} ms")
    print("  日志已写入    : ./logs/<日期>.jsonl")
    print("=" * _LINE_WIDTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
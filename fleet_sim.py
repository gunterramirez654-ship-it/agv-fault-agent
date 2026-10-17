#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fleet_sim.py —— AGV 车队运营仿真 + 分析链路压力测试
================================================================================

本脚本回答一个工程问题：
    "这套故障分析系统，放在真实运营强度下，表现如何？成本多少？瓶颈在哪？"

它分两个阶段，**两类指标来源严格分开**（这是本脚本最重要的一条纪律）：

    阶段一：运营仿真（SimPy 离散事件）
        模拟 8 台 AGV 在 4 个区域间的任务流转、电量管理、充电桩排队，
        并按各区域历史故障频率注入故障事件。
        → 产出【运营指标】：吞吐、任务周期、充电桩利用率、排队时长

    阶段二：链路压测（真实调用）
        把阶段一产生的故障事件，逐条送进真实分析链路
        （向量检索 → 规则决策 → AgentHarness → LLM），记录每次的真实开销。
        → 产出【分析指标】：召回延迟、LLM 延迟分位数、成功率、token、成本

为什么要分两阶段
--------------------------------------------------------------------------------
1) 运营指标的来源是"模型假设"，分析指标的来源是"真实测量"。混在一起讲，
   会让人误以为"充电桩利用率 82%"也是实测数据。分开后表述才准确。
2) LLM 延迟不应影响 AGV 运营——因为安全决策由规则系统给出，LLM 只做补充解读。
   这个架构优势，恰好由两阶段设计体现出来：阶段一的运营时间推进与阶段二的
   LLM 耗时完全独立。

--------------------------------------------------------------------------
模型假设（全部可配置，报告中会原样列出）
--------------------------------------------------------------------------
· 4 个作业区域：装载区 RAN / 缓存区 RAZ / 充电区 RDA / 避让区 RAB
· 区域间行驶时间见 TRAVEL_TIME
· 装卸货各 25 秒
· 每完成 1 个任务消耗 8% 电量，低于 30% 必须去充电
· 充电一次 20 分钟，充满
· 充电桩是有限资源，AGV 需排队（这是瓶颈分析的来源）
· 故障按区域以泊松过程注入，频率来自知识库中各区域案例数的相对比例
· 仿真时间与真实时间无关：24 小时仿真可在几秒内跑完（不含 LLM）

运行
--------------------------------------------------------------------------------
    # 完整跑（含真实 LLM 调用，约 2-4 分钟）
    ./run.sh sim

    # 只跑运营仿真，不调 LLM（几秒完成，用于调参）
    ./run.sh sim --no-llm

    # 自定义场景
    ./run.sh sim --hours 48 --agvs 12 --chargers 3 --seed 7

    # 限制 LLM 调用次数（控制成本）
    ./run.sh sim --max-events 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import simpy
except ImportError:
    print("[FATAL] 缺少 simpy，请执行：./venv/bin/python -m pip install simpy")
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 场景配置
# --------------------------------------------------------------------------- #
# 区域定义
ZONES: List[str] = ["region_RAN", "region_RAZ", "region_RDA", "region_RAB"]
ZONE_CN: Dict[str, str] = {
    "region_RAN": "装载区",
    "region_RAZ": "缓存区",
    "region_RDA": "充电区",
    "region_RAB": "避让区",
}

# 区域间行驶时间（秒）。双向同值，避免维护两套矩阵。
TRAVEL_TIME: Dict[Tuple[str, str], int] = {
    ("region_RAN", "region_RAZ"): 40,
    ("region_RAN", "region_RDA"): 60,
    ("region_RAN", "region_RAB"): 45,
    ("region_RAZ", "region_RDA"): 50,
    ("region_RAZ", "region_RAB"): 40,
    ("region_RDA", "region_RAB"): 55,
}

# 任务之间的待命间隔（秒，指数分布均值）
# 作用：真实工况下 AGV 并非连续作业，需要等调度派单。
#      不加这一项时 AGV 利用率会接近 100%，属于模型失真。
TASK_IDLE_MEAN = 40.0

LOAD_TIME = 25            # 装货耗时（秒）
UNLOAD_TIME = 25          # 卸货耗时（秒）
CHARGING_TIME = 20 * 60   # 充电耗时（秒）
# 电量参数标定说明：
#   原始取值 8%/任务 时，9 个任务就没电，充电频率远超真实工况，
#   导致充电桩利用率 99%、排队 39 分钟——那是模型失真，不是真实瓶颈。
#   标定后：每任务 1.5%，可用 70% 电量 ≈ 46 个任务，
#   任务周期约 2 分钟，即约 1.5 小时充一次电，符合中小型 AGV 的典型工况。
BATTERY_PER_TASK = 1.5    # 每任务耗电百分比
BATTERY_MIN = 30.0        # 低于此值必须充电
BATTERY_FULL = 100.0

# 各区域的故障权重（相对比例，来自知识库 4 区域各 4 条 → 等权）
FAULT_WEIGHTS: Dict[str, float] = {
    "region_RAN": 1.0,
    "region_RAZ": 1.0,
    "region_RDA": 1.0,
    "region_RAB": 1.0,
}

# 定价假设（元 / 百万 token）——以官方页面为准，此处仅作估算
PRICE_INPUT_PER_M = 0.5
PRICE_OUTPUT_PER_M = 2.0


# --------------------------------------------------------------------------- #
# 预置故障事件（按区域）
# --------------------------------------------------------------------------- #
# 与 web_demo.PRESET_EVENTS 保持一致，但这里只取带 area_id 的条目，
# 并按区域分组，供故障注入器随机挑选。
FAULT_EVENTS: Dict[str, List[Dict[str, str]]] = {
    "region_RAN": [
        {"label": "视觉幻视", "query": "装载区AGV把货架看错了，出现幻视导致货物位置判断错误"},
        {"label": "里程计跳变", "query": "装载区AGV定位坐标突然跳变，漂移超出允许范围"},
        {"label": "二维码丢失", "query": "装载区地面二维码识别失败，AGV停位偏差"},
        {"label": "举升过载", "query": "装载区AGV举升机构电机过载报警"},
    ],
    "region_RAZ": [
        {"label": "反光误判", "query": "缓存区地面反光，AGV以为有障碍物频繁停车"},
        {"label": "通讯丢包", "query": "缓存区AGV和调度系统通讯中断了"},
        {"label": "急停锁止", "query": "缓存区AGV突然急停锁死了"},
        {"label": "驱动过载", "query": "AGV驱动电机电流异常过载了"},
    ],
    "region_RDA": [
        {"label": "电池高温", "query": "充电时电池温度过高报警，充电过程被迫中断"},
        {"label": "充电枪异常", "query": "充电枪插上后一直充不进电，电流反复中断"},
        {"label": "BMS断连", "query": "充电时电池管理系统没有数据上传了"},
    ],
    "region_RAB": [
        {"label": "安全触边", "query": "AGV在避让区碰到东西后停下了"},
        {"label": "行人横穿", "query": "避让区有行人经过，AGV停在那里不动了"},
        {"label": "激光障碍", "query": "避让区前方有障碍物，AGV减速绕行"},
    ],
}


def travel_seconds(a: str, b: str) -> int:
    """查询两区域间的行驶时间。"""
    if a == b:
        return 0
    return TRAVEL_TIME.get((a, b)) or TRAVEL_TIME.get((b, a)) or 45


# --------------------------------------------------------------------------- #
# 阶段一：运营仿真
# --------------------------------------------------------------------------- #
@dataclass
class SimState:
    """仿真过程中累积的统计量。"""

    tasks_done: int = 0
    task_cycles: List[float] = field(default_factory=list)   # 每任务耗时
    charger_busy_seconds: float = 0.0
    charger_waits: List[float] = field(default_factory=list)
    agv_busy_seconds: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)


def agv_process(env: "simpy.Environment", name: str, chargers: "simpy.Resource",
                state: SimState, rng: random.Random) -> Any:
    """单台 AGV 的任务循环。

    一轮完整任务：装货 → 运往缓存区 → 卸货 → 返回装载区。
    任务结束后检查电量，不足则前往充电区（可能需要排队）。

    指标口径说明：
        "任务周期" = 装货 + 去程 + 卸货 + 返程，不含充电。
        充电是任务之间的独立活动，单独用充电桩利用率衡量。
    """
    # 初始电量随机化：避免所有 AGV 同时满电出发、又在同一时刻集体没电，
    # 那样会在仿真中制造出并不存在的充电排队尖峰。
    battery = rng.uniform(BATTERY_MIN, BATTERY_FULL)
    current = "region_RAN"

    while True:
        # 等待调度派单（待命，不计入忙碌时间）
        if TASK_IDLE_MEAN > 0:
            yield env.timeout(rng.expovariate(1.0 / TASK_IDLE_MEAN))

        cycle_start = env.now

        # ---- 1) 装货 ----
        yield env.timeout(LOAD_TIME)
        state.agv_busy_seconds += LOAD_TIME

        # ---- 2) 运往缓存区 ----
        t = travel_seconds(current, "region_RAZ")
        yield env.timeout(t)
        state.agv_busy_seconds += t
        current = "region_RAZ"

        # ---- 3) 卸货 ----
        yield env.timeout(UNLOAD_TIME)
        state.agv_busy_seconds += UNLOAD_TIME

        # ---- 4) 返回装载区（计入任务周期）----
        t = travel_seconds(current, "region_RAN")
        yield env.timeout(t)
        state.agv_busy_seconds += t
        current = "region_RAN"

        # ---- 5) 记账 ----
        battery -= BATTERY_PER_TASK
        state.tasks_done += 1
        state.task_cycles.append(env.now - cycle_start)

        # ---- 6) 电量管理（不计入任务周期）----
        if battery < BATTERY_MIN:
            t = travel_seconds(current, "region_RDA")
            yield env.timeout(t)
            state.agv_busy_seconds += t
            current = "region_RDA"

            wait_start = env.now
            with chargers.request() as req:
                yield req                            # ← 在此排队（不算 busy）
                state.charger_waits.append(env.now - wait_start)
                yield env.timeout(CHARGING_TIME)     # ← 占用充电桩
                state.charger_busy_seconds += CHARGING_TIME
                state.agv_busy_seconds += CHARGING_TIME
            battery = BATTERY_FULL

            t = travel_seconds(current, "region_RAN")
            yield env.timeout(t)
            state.agv_busy_seconds += t
            current = "region_RAN"


def fault_injector(env: "simpy.Environment", zone: str, rate_per_hour: float,
                   state: SimState, rng: random.Random) -> Any:
    """按泊松过程向指定区域注入故障事件。

    参数：
        rate_per_hour : 该区域平均每小时故障数
    """
    if rate_per_hour <= 0:
        return
    mean_interval = 3600.0 / rate_per_hour   # 平均间隔（秒）

    while True:
        # 指数分布间隔 = 泊松过程
        yield env.timeout(rng.expovariate(1.0 / mean_interval))
        pool = FAULT_EVENTS.get(zone) or []
        if not pool:
            continue
        ev = rng.choice(pool)
        state.events.append({
            "time_s": round(env.now, 1),
            "time_h": round(env.now / 3600.0, 2),
            "area_id": zone,
            "area_cn": ZONE_CN[zone],
            "label": ev["label"],
            "query_text": ev["query"],
        })


def run_simulation(hours: float, agv_count: int, charger_count: int,
                   fault_rate_per_agv_hour: float, seed: int) -> Dict[str, Any]:
    """运行阶段一：离散事件仿真。

    参数：
        hours                : 仿真时长（小时）
        agv_count            : AGV 数量
        charger_count        : 充电桩数量
        fault_rate_per_agv_hour : 每台 AGV 每小时故障次数
        seed                 : 随机种子（保证可复现）

    返回：
        含运营指标与事件流的字典
    """
    rng = random.Random(seed)
    env = simpy.Environment()
    chargers = simpy.Resource(env, capacity=charger_count)
    state = SimState()

    total_seconds = hours * 3600.0

    # ---- 启动 AGV 进程（错开出发，避免同时到达充电桩）----
    for i in range(agv_count):
        env.process(agv_process(env, f"AGV-{i+1:02d}", chargers, state, rng))

    # ---- 启动故障注入器（按区域权重分配总故障率）----
    total_rate = fault_rate_per_agv_hour * agv_count      # 起/小时
    weight_sum = sum(FAULT_WEIGHTS.values())
    for zone in ZONES:
        zone_rate = total_rate * FAULT_WEIGHTS[zone] / weight_sum
        env.process(fault_injector(env, zone, zone_rate, state, rng))

    # ---- 推进仿真 ----
    env.run(until=total_seconds)

    # ---- 汇总运营指标 ----
    charger_capacity_seconds = total_seconds * charger_count
    charger_util = (state.charger_busy_seconds / charger_capacity_seconds
                    if charger_capacity_seconds > 0 else 0.0)
    agv_capacity_seconds = total_seconds * agv_count
    agv_util = (state.agv_busy_seconds / agv_capacity_seconds
                if agv_capacity_seconds > 0 else 0.0)

    by_zone: Dict[str, int] = {z: 0 for z in ZONES}
    for e in state.events:
        by_zone[e["area_id"]] = by_zone.get(e["area_id"], 0) + 1

    return {
        "config": {
            "hours": hours,
            "agv_count": agv_count,
            "charger_count": charger_count,
            "fault_rate_per_agv_hour": fault_rate_per_agv_hour,
            "seed": seed,
        },
        "ops": {
            "tasks_done": state.tasks_done,
            "avg_task_cycle_s": round(statistics.fmean(state.task_cycles), 1)
                                 if state.task_cycles else 0.0,
            "agv_utilization": round(agv_util, 4),
            "charger_utilization": round(charger_util, 4),
            "avg_charger_wait_s": round(statistics.fmean(state.charger_waits), 1)
                                  if state.charger_waits else 0.0,
            "max_charger_wait_s": round(max(state.charger_waits), 1)
                                  if state.charger_waits else 0.0,
            "charge_count": len(state.charger_waits),
        },
        "faults": {
            "total": len(state.events),
            "per_hour": round(len(state.events) / hours, 2) if hours else 0,
            "by_zone": by_zone,
        },
        "events": state.events,
    }


# --------------------------------------------------------------------------- #
# 阶段二：链路压测（真实调用）
# --------------------------------------------------------------------------- #
def percentile(values: List[float], p: float) -> float:
    """最近邻法取分位数，避免插值歧义。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100.0 * len(s))) - 1))
    return s[idx]


def run_analysis(events: List[Dict[str, Any]],
                 max_events: int = 0) -> Dict[str, Any]:
    """把仿真产生的事件逐条送进真实分析链路。

    参数：
        events     : 阶段一产出的事件流
        max_events : 限制分析条数（0 表示不限制），用于控制成本

    返回：
        含分析指标与逐条明细的字典
    """
    # 延迟导入：--no-llm 模式无需加载向量模型
    import web_demo

    print("[分析] 初始化分析链路 ...", flush=True)
    t0 = time.perf_counter()
    web_demo.init_runtime()
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[分析] 初始化完成，耗时 {init_ms:.0f} ms\n", flush=True)

    targets = events[:max_events] if max_events > 0 else events
    total = len(targets)

    records: List[Dict[str, Any]] = []
    retrieval_ms: List[float] = []
    llm_ms: List[float] = []
    success = 0
    degraded = 0
    prompt_tokens = 0
    completion_tokens = 0

    print(f"[分析] 开始逐条分析，共 {total} 条 ...\n", flush=True)
    print("  " + "-" * 72)
    print(f"  {'#':>3}  {'仿真时刻':>9}  {'区域':<7} {'事件':<10} {'结论':<8} {'耗时':>8}")
    print("  " + "-" * 72)

    for i, ev in enumerate(targets, 1):
        started = time.perf_counter()
        try:
            res = web_demo.do_analyze(ev["query_text"], ev["area_id"])
            elapsed = (time.perf_counter() - started) * 1000.0
            h = res.get("harness", {})

            retrieval_ms.append(float(res.get("retrieval_cost_ms", 0)))
            llm_ms.append(float(h.get("cost_ms", 0)))
            prompt_tokens += int(h.get("prompt_tokens", 0))
            completion_tokens += int(h.get("completion_tokens", 0))
            if h.get("status") == "success":
                success += 1
            else:
                degraded += 1

            records.append({
                "index": i,
                "sim_time_h": ev["time_h"],
                "area_id": ev["area_id"],
                "label": ev["label"],
                "risk_level": res.get("risk_level"),
                "status": h.get("status"),
                "retrieval_ms": round(float(res.get("retrieval_cost_ms", 0)), 1),
                "llm_ms": round(float(h.get("cost_ms", 0)), 1),
                "tokens": int(h.get("total_tokens", 0)),
                "wall_ms": round(elapsed, 1),
            })

            mark = "✅" if h.get("status") == "success" else "⚠️"
            print(f"  {i:>3}  {ev['time_h']:>8.1f}h  {ev['area_cn']:<7} "
                  f"{ev['label']:<10} {res.get('risk_level','-'):<8} "
                  f"{h.get('cost_ms',0):>7.0f}ms {mark}", flush=True)
        except Exception as exc:  # noqa: BLE001  单条失败不应中断整轮
            degraded += 1
            print(f"  {i:>3}  {ev['time_h']:>8.1f}h  {ev['area_cn']:<7} "
                  f"{ev['label']:<10} {'失败':<8} {type(exc).__name__}", flush=True)
            records.append({
                "index": i,
                "sim_time_h": ev["time_h"],
                "area_id": ev["area_id"],
                "label": ev["label"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    print("  " + "-" * 72)

    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens / 1e6) * PRICE_INPUT_PER_M + \
           (completion_tokens / 1e6) * PRICE_OUTPUT_PER_M
    hours = records[-1]["sim_time_h"] if records and "sim_time_h" in records[-1] else 0.0

    return {
        "analyzed": total,
        "success": success,
        "degraded": degraded,
        "success_rate": round(success / total, 4) if total else 0.0,
        "init_ms": round(init_ms, 1),
        "retrieval_p50_ms": round(percentile(retrieval_ms, 50), 1),
        "retrieval_p95_ms": round(percentile(retrieval_ms, 95), 1),
        "llm_p50_ms": round(percentile(llm_ms, 50), 1),
        "llm_p95_ms": round(percentile(llm_ms, 95), 1),
        "llm_p99_ms": round(percentile(llm_ms, 99), 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_yuan": round(cost, 4),
        "sim_hours_covered": round(hours, 2),
        "records": records,
    }


# --------------------------------------------------------------------------- #
# What-if 对比：让仿真成为决策工具
# --------------------------------------------------------------------------- #
def run_what_if(hours: float, agvs: int, charger_options: List[int],
                fault_rate: float, seed: int) -> List[Dict[str, Any]]:
    """对不同的充电桩数量分别仿真，对比运营效果。

    这是仿真最有价值的用途：不是"算出一个数"，而是回答
    "如果多加一个充电桩，能带来多少改善"。

    参数：
        charger_options : 要对比的充电桩数量列表，例如 [1, 2, 3]

    返回：
        每个配置的指标字典列表
    """
    rows: List[Dict[str, Any]] = []
    for n in charger_options:
        sim = run_simulation(
            hours=hours,
            agv_count=agvs,
            charger_count=n,
            fault_rate_per_agv_hour=fault_rate,
            seed=seed,          # 固定种子：保证不同配置的故障流一致，可公平对比
        )
        ops = sim["ops"]
        rows.append({
            "chargers": n,
            "tasks_done": ops["tasks_done"],
            "avg_task_cycle_s": ops["avg_task_cycle_s"],
            "charger_utilization": ops["charger_utilization"],
            "avg_wait_s": ops["avg_charger_wait_s"],
            "max_wait_s": ops["max_charger_wait_s"],
            "agv_utilization": ops["agv_utilization"],
        })
    return rows


def print_what_if(rows: List[Dict[str, Any]], hours: float, agvs: int) -> None:
    """打印 what-if 对比表。"""
    if not rows:
        return

    print("\n" + _line())
    print("  What-if 对比：充电桩数量对运营的影响")
    print(_line())
    print(f"  固定条件：{hours:.0f} 小时 / {agvs} 台 AGV / 同一随机种子（故障流一致）\n")

    print(f"  {'充电桩':>6}  {'完成任务':>9}  {'充电桩利用率':>12}  "
          f"{'平均排队':>10}  {'最长排队':>10}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['chargers']:>6}  {r['tasks_done']:>9}  "
              f"{r['charger_utilization']:>11.1%}  "
              f"{r['avg_wait_s']/60:>9.1f}分  "
              f"{r['max_wait_s']/60:>9.1f}分")
    print("  " + "-" * 68)

    # ---- 相邻配置的边际收益 ----
    print("\n  每增加一个充电桩的边际收益：")
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        d_tasks = cur["tasks_done"] - prev["tasks_done"]
        pct = d_tasks / prev["tasks_done"] * 100 if prev["tasks_done"] else 0.0
        d_wait = prev["avg_wait_s"] / 60 - cur["avg_wait_s"] / 60
        print(f"    {prev['chargers']} → {cur['chargers']} 个："
              f"吞吐 {pct:+5.1f}%，"
              f"平均排队减少 {d_wait:.1f} 分钟，"
              f"利用率 {prev['charger_utilization']:.0%} → "
              f"{cur['charger_utilization']:.0%}")

    # ---- 拐点检测：边际吞吐增益首次低于 1% 即视为收益递减 ----
    # 理由：多一个充电桩意味着设备与维护成本，吞吐提升不足 1% 通常不划算。
    knee = rows[0]["chargers"]
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        gain = ((cur["tasks_done"] - prev["tasks_done"]) / prev["tasks_done"]
                if prev["tasks_done"] else 0.0)
        if gain >= 0.01:
            knee = cur["chargers"]
        else:
            break

    print("\n  建议：")
    print(f"    · 推荐配置为 {knee} 个充电桩"
          f"（完成任务 {next(r['tasks_done'] for r in rows if r['chargers'] == knee)} 个）")
    for r in rows:
        if r["charger_utilization"] > 0.85:
            print(f"    · {r['chargers']} 个充电桩时利用率 "
                  f"{r['charger_utilization']:.0%}，已接近饱和，"
                  f"平均排队 {r['avg_wait_s']/60:.1f} 分钟")
    beyond = [r for r in rows if r["chargers"] > knee]
    if beyond:
        r = beyond[0]
        print(f"    · 超过 {knee} 个后收益递减"
              f"（{knee} → {r['chargers']} 个仅提升 "
              f"{(r['tasks_done'] - next(x['tasks_done'] for x in rows if x['chargers'] == knee)) / next(x['tasks_done'] for x in rows if x['chargers'] == knee) * 100:+.1f}%），"
              f"不建议继续投入")


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def _line(ch: str = "=", n: int = 74) -> str:
    return ch * n


def print_report(sim: Dict[str, Any], ana: Optional[Dict[str, Any]]) -> None:
    """打印运营仿真报告。"""
    cfg = sim["config"]
    ops = sim["ops"]
    faults = sim["faults"]

    print("\n" + _line())
    print("  AGV 车队运营仿真报告")
    print(_line())

    print("\n【仿真配置】（以下均为模型假设，非实测）")
    print(f"  仿真时长        : {cfg['hours']:.1f} 小时")
    print(f"  AGV 数量        : {cfg['agv_count']} 台")
    print(f"  充电桩数量      : {cfg['charger_count']} 个")
    print(f"  故障率          : 每台 AGV 每小时 {cfg['fault_rate_per_agv_hour']} 起")
    print(f"  随机种子        : {cfg['seed']}（固定种子保证可复现）")

    print("\n【运营指标】（离散事件仿真，SimPy）")
    print(f"  完成任务数      : {ops['tasks_done']} 个")
    print(f"  平均任务周期    : {ops['avg_task_cycle_s']/60:.1f} 分钟")
    print(f"  AGV 利用率      : {ops['agv_utilization']:.1%}")
    print(f"  充电桩利用率    : {ops['charger_utilization']:.1%}")
    print(f"  充电次数        : {ops['charge_count']} 次")
    print(f"  平均充电等待    : {ops['avg_charger_wait_s']/60:.1f} 分钟")
    print(f"  最长充电等待    : {ops['max_charger_wait_s']/60:.1f} 分钟")

    print("\n【故障流】（按区域泊松过程注入）")
    print(f"  事件总数        : {faults['total']} 起")
    print(f"  故障率          : {faults['per_hour']:.2f} 起/小时")
    parts = [f"{ZONE_CN[z]} {n}" for z, n in faults["by_zone"].items()]
    print(f"  区域分布        : " + " / ".join(parts))

    if ana:
        print("\n【分析链路】（真实调用，实测）")
        print(f"  分析条数        : {ana['analyzed']} 条")
        print(f"  成功率          : {ana['success_rate']:.1%} "
              f"({ana['success']}/{ana['analyzed']})")
        if ana["degraded"]:
            print(f"  降级次数        : {ana['degraded']} 次（均返回规则结论）")
        print(f"  召回延迟        : P50 {ana['retrieval_p50_ms']:.1f}ms / "
              f"P95 {ana['retrieval_p95_ms']:.1f}ms")
        print(f"  LLM 延迟        : P50 {ana['llm_p50_ms']/1000:.2f}s / "
              f"P95 {ana['llm_p95_ms']/1000:.2f}s / "
              f"P99 {ana['llm_p99_ms']/1000:.2f}s")
        print(f"  Token 消耗      : 输入 {ana['prompt_tokens']:,} / "
              f"输出 {ana['completion_tokens']:,} / "
              f"合计 {ana['total_tokens']:,}")
        print(f"  本次成本        : ¥{ana['cost_yuan']:.4f}"
              f"（按输入 ¥{PRICE_INPUT_PER_M}/M、输出 ¥{PRICE_OUTPUT_PER_M}/M 估算）")

        # ---- 年化估算 ----
        if ana["sim_hours_covered"] > 0:
            cost_per_hour = ana["cost_yuan"] / ana["sim_hours_covered"]
            yearly = cost_per_hour * 24 * 365
            print(f"  年化成本        : ≈ ¥{yearly:,.0f}"
                  f"（按 {ana['sim_hours_covered']:.1f} 小时样本外推）")

    # ---- 瓶颈分析 ----
    print("\n【瓶颈与建议】")
    findings: List[str] = []

    cu = ops["charger_utilization"]
    if cu > 0.75:
        findings.append(
            f"充电桩利用率 {cu:.0%}，已接近饱和；"
            f"建议评估扩容至 {cfg['charger_count'] + 1} 个")
    elif cu > 0.5:
        findings.append(f"充电桩利用率 {cu:.0%}，处于合理区间偏上，可持续观察")
    else:
        findings.append(f"充电桩利用率 {cu:.0%}，资源充足")

    if ops["max_charger_wait_s"] > 900:
        findings.append(
            f"最长充电等待 {ops['max_charger_wait_s']/60:.1f} 分钟，"
            f"存在任务延误风险")

    if ana and ana["llm_p99_ms"] > 5000:
        findings.append(
            f"LLM P99 延迟 {ana['llm_p99_ms']/1000:.1f}s，"
            f"建议对超长报告设置更严格的 max_tokens")

    if ana and ana["success_rate"] < 0.99:
        findings.append(
            f"LLM 成功率 {ana['success_rate']:.1%}，"
            f"降级均返回规则结论，未丢失安全决策")

    # 区域故障集中度
    total_faults = sum(faults["by_zone"].values())
    if total_faults:
        top_zone = max(faults["by_zone"].items(), key=lambda kv: kv[1])
        ratio = top_zone[1] / total_faults
        if ratio > 0.32:
            findings.append(
                f"{ZONE_CN[top_zone[0]]}故障占比 {ratio:.0%}，"
                f"建议增加该区域巡检频次")

    for f in findings:
        print(f"  · {f}")

    print("\n" + _line())
    print("  说明：运营指标来自仿真模型；分析链路的延迟与成本为真实调用实测。")
    print(_line())


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="AGV 车队运营仿真 + 分析链路压力测试")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="仿真时长（小时），默认 24")
    parser.add_argument("--agvs", type=int, default=8,
                        help="AGV 数量，默认 8")
    parser.add_argument("--chargers", type=int, default=2,
                        help="充电桩数量，默认 2")
    parser.add_argument("--fault-rate", type=float, default=0.25,
                        help="每台 AGV 每小时故障次数，默认 0.25（即每 4 小时一次）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--no-llm", action="store_true",
                        help="只跑运营仿真，不调用 LLM")
    parser.add_argument("--max-events", type=int, default=0,
                        help="限制分析条数（0=不限），用于控制 LLM 成本")
    parser.add_argument("--json-out", default="",
                        help="把完整结果写入 JSON 文件")
    parser.add_argument("--what-if", action="store_true",
                        help="对比不同充电桩数量（1/2/3）的运营效果，不调用 LLM")
    args = parser.parse_args()

    print(_line())
    print("  AGV 车队运营仿真 + 分析链路压力测试")
    print(_line())

    # ---- What-if 模式：只做运营对比，不调 LLM ----
    if args.what_if:
        print(f"\n[What-if] 对比充电桩数量对运营的影响 "
              f"({args.hours:.0f}h / {args.agvs} 台 AGV) ...", flush=True)
        rows = run_what_if(
            hours=args.hours,
            agvs=args.agvs,
            charger_options=[1, 2, 3],
            fault_rate=args.fault_rate,
            seed=args.seed,
        )
        print_what_if(rows, args.hours, args.agvs)
        print("\n" + _line())
        return 0

    # ---- 阶段一 ----
    print(f"\n[阶段一] 运营仿真 "
          f"({args.hours:.0f}h / {args.agvs} 台 AGV / {args.chargers} 个充电桩) ...",
          flush=True)
    t0 = time.perf_counter()
    sim = run_simulation(
        hours=args.hours,
        agv_count=args.agvs,
        charger_count=args.chargers,
        fault_rate_per_agv_hour=args.fault_rate,
        seed=args.seed,
    )
    sim_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[阶段一] 完成，耗时 {sim_ms:.0f} ms，"
          f"产生 {sim['faults']['total']} 起故障事件")

    # ---- 阶段二 ----
    ana = None
    if args.no_llm:
        print("\n[阶段二] 已跳过（--no-llm）")
    elif not sim["events"]:
        print("\n[阶段二] 没有故障事件，跳过分析")
    else:
        print()
        ana = run_analysis(sim["events"], max_events=args.max_events)

    # ---- 报告 ----
    print_report(sim, ana)

    # ---- 落盘 ----
    if args.json_out:
        payload = {"simulation": sim, "analysis": ana}
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"\n[输出] 完整结果已写入：{args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

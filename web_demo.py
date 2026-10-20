#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_demo.py —— AGV 故障分析可视化演示服务（V1 POC）
================================================================================

用途
--------------------------------------------------------------------------------
为 AGV 故障分析链路提供一个本地 Web 演示界面，用于现场演示与联调。
不引入任何第三方 Web 框架，仅使用 Python 标准库 http.server，零额外依赖。

界面能力
--------------------------------------------------------------------------------
· 2D 工厂俯视图：四个作业区域（装载/缓存/充电/避让），AGV 沿区域移动
· 一键注入预置现场事件，驱动真实链路：向量检索 → 规则决策 → LLM 分析
· 实时展示：召回 Top3、相似度、风险等级、处置动作、LLM 报告、Harness 指标
· 底部日志流：完整记录每次交互

----------------------------------------------------------------------------
【关键设计】快慢分离的可视化
----------------------------------------------------------------------------
检索与规则决策约 10ms，LLM 分析约 2s。接口因此拆成两个：

    POST /api/retrieve  → 检索 + 规则决策（毫秒级，立即返回）
    POST /api/analyze   → LLM 辅助分析（秒级，单独请求）

前端先显示检索结果与安全结论，再异步等待 LLM 报告。
这直观体现了架构特性：安全控制不等待 LLM，LLM 只补充解读。

接口
----------------------------------------------------------------------------
    GET  /                → 返回演示界面
    GET  /api/config      → 预置事件、区域、知识库概况
    POST /api/retrieve    → {"query_text": str, "area_id": str|null}
                            → 召回结果 + max_sim/avg_similarity + 风险决策
    POST /api/analyze     → {"query_text": str, "area_id": str|null}
                            → LLM 报告 + Harness 指标 + 风险决策

运行
----------------------------------------------------------------------------
    cd agv_embodied_project
    ./venv/bin/python web_demo.py
    浏览器打开 http://127.0.0.1:8000

    # 端口被占用时可指定：
    WEB_DEMO_PORT=8080 ./venv/bin/python web_demo.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 项目模块
# --------------------------------------------------------------------------- #
try:
    from agent_harness import AgentHarness
    from vector_store import KnowledgeBaseError, VectorStore
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] 项目模块导入失败：{exc}")
    print("        请确认在 agv_embodied_project 目录下运行本脚本。")
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 运营仿真模块（可选：缺 simpy 时页签会提示未安装）
# --------------------------------------------------------------------------- #
try:
    from fleet_sim import run_simulation, run_what_if
    _SIM_OK = True
except Exception as _sim_exc:  # noqa: BLE001
    _SIM_OK = False
    _SIM_ERR = f"{type(_sim_exc).__name__}: {_sim_exc}"
    run_simulation = None       # type: ignore[assignment]
    run_what_if = None          # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_PATH = os.path.join(_MODULE_DIR, "templates", "index.html")

HOST = os.getenv("WEB_DEMO_HOST", "127.0.0.1")
PORT = int(os.getenv("WEB_DEMO_PORT", "8000"))

# 规则阈值（与 test_vector_store.py 保持一致，便于演示结果可对照）
NO_MATCH_MAX_THRESHOLD = 0.45
MID_RISK_MAX_THRESHOLD = 0.67
HIGH_RISK_MAX_THRESHOLD = 0.78

# 预置现场事件：覆盖 4 个作业区域 + 2 个"区域未知"场景
# 每条事件对应知识库中真实存在的故障类型，用于演示检索质量
PRESET_EVENTS: List[Dict[str, Any]] = [
    # ---------------- 装载区 region_RAN ----------------
    {
        "id": "ran_vision",
        "group": "装载区",
        "label": "视觉幻视",
        "area_id": "region_RAN",
        "query_text": "装载区AGV把货架看错了，出现幻视导致货物位置判断错误",
    },
    {
        "id": "ran_odom",
        "group": "装载区",
        "label": "里程计跳变",
        "area_id": "region_RAN",
        "query_text": "装载区AGV定位坐标突然跳变，漂移超出允许范围",
    },
    {
        "id": "ran_qr",
        "group": "装载区",
        "label": "二维码丢失",
        "area_id": "region_RAN",
        "query_text": "装载区地面二维码识别失败，AGV停位偏差",
    },
    {
        "id": "ran_lift",
        "group": "装载区",
        "label": "举升过载",
        "area_id": "region_RAN",
        "query_text": "装载区AGV举升机构电机过载报警",
    },
    # ---------------- 缓存区 region_RAZ ----------------
    {
        "id": "raz_reflect",
        "group": "缓存区",
        "label": "反光误判",
        "area_id": "region_RAZ",
        "query_text": "缓存区地面反光，AGV以为有障碍物频繁停车",
    },
    {
        "id": "raz_wifi",
        "group": "缓存区",
        "label": "通讯丢包",
        "area_id": "region_RAZ",
        "query_text": "缓存区AGV和调度系统通讯中断了",
    },
    {
        "id": "raz_estop",
        "group": "缓存区",
        "label": "急停锁止",
        "area_id": "region_RAZ",
        "query_text": "缓存区AGV突然急停锁死了",
    },
    {
        "id": "raz_overload",
        "group": "缓存区",
        "label": "驱动过载",
        "area_id": "region_RAZ",
        "query_text": "AGV驱动电机电流异常过载了",
    },
    # ---------------- 充电区 region_RDA ----------------
    {
        "id": "rda_battery",
        "group": "充电区",
        "label": "电池高温",
        "area_id": "region_RDA",
        "query_text": "充电时电池温度过高报警，充电过程被迫中断",
    },
    {
        "id": "rda_gun",
        "group": "充电区",
        "label": "充电枪异常",
        "area_id": "region_RDA",
        "query_text": "充电枪插上后一直充不进电，电流反复中断",
    },
    {
        "id": "rda_bms",
        "group": "充电区",
        "label": "BMS断连",
        "area_id": "region_RDA",
        "query_text": "充电时电池管理系统没有数据上传了",
    },
    # ---------------- 避让区 region_RAB ----------------
    {
        "id": "rab_touch",
        "group": "避让区",
        "label": "安全触边",
        "area_id": "region_RAB",
        "query_text": "AGV在避让区碰到东西后停下了",
    },
    {
        "id": "rab_person",
        "group": "避让区",
        "label": "行人横穿",
        "area_id": "region_RAB",
        "query_text": "避让区有行人经过，AGV停在那里不动了",
    },
    {
        "id": "rab_laser",
        "group": "避让区",
        "label": "激光障碍",
        "area_id": "region_RAB",
        "query_text": "避让区前方有障碍物，AGV减速绕行",
    },
    # ---------------- 区域未知（演示全局召回）----------------
    {
        "id": "unknown_touch",
        "group": "区域未知",
        "label": "碰撞停车",
        "area_id": None,
        "query_text": "AGV撞到东西后触发安全触边，车停住了",
    },
    {
        "id": "unknown_alert",
        "group": "区域未知",
        "label": "不明报警",
        "area_id": None,
        "query_text": "AGV突然报警停机，现场暂时没查到原因",
    },
]

# 区域展示名（前端渲染用）
AREA_LABELS: Dict[str, str] = {
    "region_RAN": "装载区",
    "region_RAZ": "缓存区",
    "region_RDA": "充电区",
    "region_RAB": "避让区",
}


# --------------------------------------------------------------------------- #
# 全局单例（启动时加载一次，避免每次请求重复向量化）
# --------------------------------------------------------------------------- #
_STORE: Optional[VectorStore] = None
_HARNESS: Optional[AgentHarness] = None
# LLM 调用串行化：演示服务并发场景下避免互相抢占与统计错乱
_LLM_LOCK = threading.Lock()


def init_runtime() -> None:
    """初始化向量库与 LLM 外壳（仅在服务启动时执行一次）。"""
    global _STORE, _HARNESS

    print("[init] 正在加载知识库与向量模型 ...", flush=True)
    t0 = time.perf_counter()
    _STORE = VectorStore()
    init_ms = (time.perf_counter() - t0) * 1000.0

    # 输出预算由 Harness 统一管控（详见 agent_harness 模块说明）
    _HARNESS = AgentHarness(
        max_retries=2,
        retry_base_delay=1.0,
        max_output_tokens=600,
        max_report_chars=800,
    )

    print(f"[init] 完成：{_STORE.count()} 条案例，"
          f"{len(_STORE.records[0].embedding)} 维向量，耗时 {init_ms:.0f} ms")
    print(f"[init] 区域分布：{_STORE.area_distribution()}")


# --------------------------------------------------------------------------- #
# 业务逻辑
# --------------------------------------------------------------------------- #
def decide_risk(max_sim: float) -> Dict[str, str]:
    """按硬编码安全规则给出风险等级与处置动作。

    判据使用最高相似度 max_sim（理由见 test_vector_store.decide_risk 的说明）。

    参数：
        max_sim : 召回结果中的最高相似度

    返回：
        含 risk_level / action / decision 的字典
    """
    if max_sim < NO_MATCH_MAX_THRESHOLD:
        return {
            "risk_level": "高风险",
            "action": "AGV减速，重新扫描环境，上传异常日志",
            "decision": "无有效匹配 → 高风险，减速重扫",
        }
    if max_sim >= HIGH_RISK_MAX_THRESHOLD:
        return {
            "risk_level": "高风险",
            "action": "AGV立即暂停，人工确认现场",
            "decision": "高匹配 → 高风险，立即暂停",
        }
    if max_sim >= MID_RISK_MAX_THRESHOLD:
        return {
            "risk_level": "中风险",
            "action": "AGV减速运行，二次校验传感器数据",
            "decision": "中等匹配 → 中风险，减速校验",
        }
    return {
        "risk_level": "低风险",
        "action": "AGV继续正常作业",
        "decision": "弱相关 → 低风险，正常作业",
    }


def do_retrieve(query_text: str, area_id: Optional[str]) -> Dict[str, Any]:
    """执行召回 + 规则决策（毫秒级），返回结构化结果。"""
    assert _STORE is not None

    t0 = time.perf_counter()
    results = _STORE.search(query_text=query_text, top_k=3, area_id=area_id)
    cost_ms = (time.perf_counter() - t0) * 1000.0

    if results:
        scores = [float(x["score"]) for x in results]
        max_sim = max(scores)
        avg_sim = sum(scores) / len(scores)
    else:
        max_sim = 0.0
        avg_sim = 0.0

    risk = decide_risk(max_sim)

    return {
        "ok": True,
        "query_text": query_text,
        "area_id": area_id,
        "results": results,
        "max_sim": round(max_sim, 4),
        "avg_similarity": round(avg_sim, 4),
        "cost_ms": round(cost_ms, 1),
        **risk,
    }


def do_analyze(query_text: str, area_id: Optional[str]) -> Dict[str, Any]:
    """执行完整链路：召回 → 规则决策 → LLM 分析（秒级）。"""
    assert _STORE is not None
    assert _HARNESS is not None

    # 先完成召回与决策（快），再交给 LLM 做解读（慢）
    base = do_retrieve(query_text, area_id)

    with _LLM_LOCK:
        llm_result = _HARNESS.analyze(
            query_text=query_text,
            search_results=base["results"],
            rule_risk=base["risk_level"],
            rule_action=base["action"],
        )

    return {
        "ok": True,
        "query_text": query_text,
        "area_id": area_id,
        "risk_level": base["risk_level"],
        "action": base["action"],
        "decision": base["decision"],
        "max_sim": base["max_sim"],
        "avg_similarity": base["avg_similarity"],
        "retrieval_cost_ms": base["cost_ms"],
        "report": llm_result.report,
        "harness": llm_result.to_dict(),
    }


# --------------------------------------------------------------------------- #
# HTTP 处理
# --------------------------------------------------------------------------- #
class DemoHandler(BaseHTTPRequestHandler):
    """演示服务的请求处理器。"""

    # 静默默认的访问日志，避免刷屏；只打印错误
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        """返回 JSON 响应。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str, status: int = 200) -> None:
        """返回 HTML 响应。"""
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        """读取并解析请求体 JSON（失败返回空字典）。"""
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------ #
    # 路由
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._serve_index()
        elif self.path == "/api/config":
            self._serve_config()
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/retrieve":
            self._handle_retrieve()
        elif self.path == "/api/analyze":
            self._handle_analyze()
        elif self.path == "/api/simulate":
            self._handle_simulate()
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)

    # ------------------------------------------------------------------ #
    # 各接口实现
    # ------------------------------------------------------------------ #
    def _serve_index(self) -> None:
        """返回演示界面。"""
        if not os.path.isfile(_TEMPLATE_PATH):
            self._send_html(
                "<h1>界面文件缺失</h1>"
                f"<p>未找到：{_TEMPLATE_PATH}</p>",
                status=500,
            )
            return
        try:
            with open(_TEMPLATE_PATH, "r", encoding="utf-8") as fp:
                self._send_html(fp.read())
        except OSError as exc:
            self._send_html(f"<h1>界面读取失败</h1><pre>{exc}</pre>", status=500)

    def _serve_config(self) -> None:
        """返回预置事件与知识库概况。"""
        assert _STORE is not None
        self._send_json({
            "ok": True,
            "events": PRESET_EVENTS,
            "area_labels": AREA_LABELS,
            "kb_count": _STORE.count(),
            "area_distribution": _STORE.area_distribution(),
            "embedding_dim": len(_STORE.records[0].embedding),
            "thresholds": {
                "no_match": NO_MATCH_MAX_THRESHOLD,
                "mid": MID_RISK_MAX_THRESHOLD,
                "high": HIGH_RISK_MAX_THRESHOLD,
            },
        })

    def _handle_retrieve(self) -> None:
        """召回 + 规则决策接口（毫秒级）。"""
        data = self._read_json()
        query = str(data.get("query_text", "") or "").strip()
        area = data.get("area_id")

        if not query:
            self._send_json({"ok": False, "error": "query_text 不能为空"}, status=400)
            return

        try:
            self._send_json(do_retrieve(query, area))
        except KnowledgeBaseError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500
            )

    def _handle_analyze(self) -> None:
        """完整链路接口（含 LLM，秒级）。"""
        data = self._read_json()
        query = str(data.get("query_text", "") or "").strip()
        area = data.get("area_id")

        if not query:
            self._send_json({"ok": False, "error": "query_text 不能为空"}, status=400)
            return

        try:
            self._send_json(do_analyze(query, area))
        except KnowledgeBaseError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500
            )


    def _handle_simulate(self) -> None:
        """运营仿真接口：跑 SimPy 模型，返回指标 + what-if 对比。

        纯计算，不调用 LLM，通常几十毫秒到几百毫秒返回。
        """
        if not _SIM_OK:
            self._send_json({
                "ok": False,
                "error": f"运营仿真不可用（缺少依赖）：{_SIM_ERR}",
            }, status=500)
            return

        data = self._read_json()
        try:
            hours = float(data.get("hours", 24))
            agvs = int(data.get("agvs", 8))
            chargers = int(data.get("chargers", 2))
            fault_rate = float(data.get("fault_rate", 0.25))
            seed = int(data.get("seed", 42))
        except (TypeError, ValueError):
            self._send_json({"ok": False, "error": "参数格式错误"}, status=400)
            return

        # 参数边界保护，避免误输入把服务卡死
        hours = max(1.0, min(hours, 168.0))
        agvs = max(1, min(agvs, 40))
        chargers = max(1, min(chargers, 20))
        fault_rate = max(0.0, min(fault_rate, 2.0))

        try:
            sim = run_simulation(hours=hours, agv_count=agvs,
                                 charger_count=chargers,
                                 fault_rate_per_agv_hour=fault_rate,
                                 seed=seed)
            compare = run_what_if(hours=hours, agvs=agvs,
                                  charger_options=[1, 2, 3],
                                  fault_rate=fault_rate, seed=seed)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                status=500)
            return

        # 事件按小时分桶，供前端画时间线
        buckets = [0] * int(math.ceil(hours))
        for e in sim["events"]:
            idx = min(int(e["time_h"]), len(buckets) - 1)
            if idx >= 0:
                buckets[idx] += 1

        self._send_json({
            "ok": True,
            "config": sim["config"],
            "ops": sim["ops"],
            "faults": sim["faults"],
            "events_per_hour": buckets,
            "what_if": compare,
            "area_labels": AREA_LABELS,
        })


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    """启动演示服务。"""
    print("=" * 72)
    print("  AGV 故障分析可视化演示服务（V1 POC）")
    print("=" * 72)

    init_runtime()

    # 端口占用时自动尝试下一个
    global PORT
    server = None
    for candidate in (PORT, PORT + 1, PORT + 2):
        try:
            server = ThreadingHTTPServer((HOST, candidate), DemoHandler)
            PORT = candidate
            break
        except OSError:
            print(f"[warn] 端口 {candidate} 被占用，尝试下一个 ...")

    if server is None:
        print("[FATAL] 没有可用端口，请用 WEB_DEMO_PORT 指定其它端口。")
        return 1

    url = f"http://{HOST}:{PORT}"
    print()
    print(f"  服务已启动：{url}")
    print("  按 Ctrl+C 停止")
    print("=" * 72)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[exit] 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

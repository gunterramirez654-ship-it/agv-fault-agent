#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py —— 全功能验证
================================================================================

一条命令跑完所有功能验证，输出一张通过/失败表，便于交付前自检或演示前确认。

验证项：
    1. 环境与依赖（解释器、包、模型文件、.env）
    2. 知识库与向量化（16 条 / 512 维）
    3. 召回与规则决策（3 组用例，检查 top1 命中与风险等级）
    4. LLM 分析链路（1 次真实调用，经 Harness 校验）
    5. Web 接口（config / retrieve / analyze / simulate）
    6. 3D 联动（可选，CoppeliaSim 未启动时只提示）

运行：
    cd agv_embodied_project
    ./run.sh verify
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PASS, FAIL, WARN = "✅", "❌", "⚠️ "

# 验证结果：(编号, 名称, 状态, 详情)
RESULTS: List[Tuple[str, str, str, str]] = []

# Web 验证使用的临时端口，避免和正在运行的服务冲突
WEB_PORT = 8111


def add(no: str, name: str, ok: Optional[bool], detail: str = "") -> None:
    """记录一项结果。ok=None 表示可选/跳过。"""
    status = PASS if ok is True else (FAIL if ok is False else WARN)
    RESULTS.append((no, name, status, detail))
    print(f"  {status}  [{no}] {name}" + (f"  —— {detail}" if detail else ""))


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 1. 环境
# --------------------------------------------------------------------------- #
def check_env() -> bool:
    print("\n[1/6] 环境与依赖")
    ok = True

    in_venv = sys.prefix != sys.base_prefix
    add("1.1", "使用虚拟环境解释器", in_venv,
        os.path.basename(sys.executable) if in_venv else "请用 ./run.sh")
    ok &= in_venv

    import importlib.util
    for mod in ("chromadb", "onnxruntime", "tokenizers", "numpy", "openai"):
        has = importlib.util.find_spec(mod) is not None
        add(f"1.2", f"依赖 {mod}", has)
        ok &= has

    model_dir = os.getenv("BGE_MODEL_DIR") or "onnx_models/bge-small-zh-v1.5/onnx"
    need = ["model.onnx", "tokenizer.json", "config.json"]
    missing = [f for f in need if not os.path.isfile(os.path.join(model_dir, f))]
    add("1.3", "BGE 向量模型文件", not missing,
        "缺失 " + ", ".join(missing) if missing else f"{model_dir}")
    ok &= not missing

    has_env = os.path.isfile(".env")
    add("1.4", ".env 配置", has_env, "" if has_env else "cp .env.example .env")
    ok &= has_env
    return ok


# --------------------------------------------------------------------------- #
# 2~4. 核心链路（直接调模块，不走 HTTP）
# --------------------------------------------------------------------------- #
def check_core() -> bool:
    print("\n[2/6] 知识库与向量化")
    from vector_store import VectorStore

    t0 = time.perf_counter()
    vs = VectorStore()
    init_ms = (time.perf_counter() - t0) * 1000

    count = vs.count()
    dim = len(vs.records[0].embedding) if vs.records else 0
    dist = vs.area_distribution()

    ok = (count == 16 and dim == 512 and len(dist) == 4)
    add("2.1", "知识库加载", ok, f"{count} 条 / {dim} 维 / {init_ms:.0f} ms")
    add("2.2", "四区域分布均衡", len(dist) == 4 and all(v == 4 for v in dist.values()),
        str(dist))
    return ok


def check_retrieval(vs: Any) -> bool:
    print("\n[3/6] 召回与规则决策")
    from test_vector_store import decide_risk

    cases = [
        ("装载区AGV把货架看错了，出现幻视导致货物位置判断错误",
         "region_RAN", "fault_RAN_001"),
        ("充电时电池温度过高报警，充电过程被迫中断",
         "region_RDA", "fault_RDA_001"),
        ("AGV撞到东西后触发安全触边，车停住了",
         "region_RAB", "fault_RAB_002"),
    ]

    hits = 0
    for q, area, expect in cases:
        res = vs.search(q, top_k=3, area_id=area)
        if not res:
            add("3.x", f"召回「{q[:12]}…」", False, "无结果")
            continue
        top1 = res[0]
        max_sim = max(float(r["score"]) for r in res)
        risk, action, _ = decide_risk(max_sim, 0.0)
        hit = top1["fault_id"] == expect
        hits += int(hit)
        add("3.1", f"召回「{q[:10]}…」", hit,
            f"top1={top1['fault_id']} ({max_sim:.4f}) → {risk}")

    add("3.2", "召回命中率", hits == len(cases), f"{hits}/{len(cases)}")
    return hits == len(cases)


def check_llm(vs: Any) -> bool:
    print("\n[4/6] LLM 分析链路")
    from agent_harness import AgentHarness

    q = "装载区AGV把货架看错了，出现幻视导致货物位置判断错误"
    res = vs.search(q, top_k=3, area_id="region_RAN")

    t0 = time.perf_counter()
    harness = AgentHarness(max_retries=2, retry_base_delay=1.0,
                           max_output_tokens=600, max_report_chars=800)
    r = harness.analyze(q, res, "高风险", "AGV立即暂停，人工确认现场")
    ms = (time.perf_counter() - t0) * 1000

    add("4.1", "LLM 调用成功", r.is_success,
        f"{r.status} / {ms:.0f} ms / {r.total_tokens} tokens")
    add("4.2", "输出通过安全校验", r.validation.passed,
        r.validation.summary)
    add("4.3", "三段结构完整", not r.validation.missing_sections,
        "缺 " + str(r.validation.missing_sections) if r.validation.missing_sections else "一/二/三 齐全")
    add("4.4", "无控制类违规措辞", not r.validation.violations,
        str(r.validation.violations) if r.validation.violations else "0 条")
    return r.is_success and r.validation.passed


# --------------------------------------------------------------------------- #
# 5. Web 接口（起一个临时服务实测）
# --------------------------------------------------------------------------- #
def check_web() -> bool:
    print(f"\n[5/6] Web 接口（临时端口 {WEB_PORT}）")

    env = os.environ.copy()
    env["WEB_DEMO_PORT"] = str(WEB_PORT)
    proc = subprocess.Popen(
        [sys.executable, "web_demo.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    try:
        # 等服务就绪
        base = f"http://127.0.0.1:{WEB_PORT}"
        ready = False
        for _ in range(40):
            if port_open(WEB_PORT):
                ready = True
                break
            time.sleep(0.5)
        add("5.0", "服务启动", ready, base)
        if not ready:
            return False

        def get_json(path: str) -> Dict[str, Any]:
            with urllib.request.urlopen(base + path, timeout=15) as fp:
                return json.loads(fp.read().decode("utf-8"))

        def post_json(path: str, payload: Dict[str, Any],
                      timeout: int = 40) -> Dict[str, Any]:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                base + path, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as fp:
                return json.loads(fp.read().decode("utf-8"))

        # 5.1 配置
        cfg = get_json("/api/config")
        add("5.1", "GET /api/config", bool(cfg.get("ok")),
            f"{cfg.get('kb_count')} 条 / {len(cfg.get('events', []))} 个事件")

        # 5.2 召回
        r1 = post_json("/api/retrieve",
                       {"query_text": "充电时电池温度过高报警",
                        "area_id": "region_RDA"})
        add("5.2", "POST /api/retrieve", bool(r1.get("ok")),
            f"{r1.get('risk_level')} / max_sim={r1.get('max_sim')} / "
            f"{r1.get('cost_ms')} ms")

        # 5.3 完整分析
        r2 = post_json("/api/analyze",
                       {"query_text": "充电时电池温度过高报警",
                        "area_id": "region_RDA"})
        h = r2.get("harness", {})
        add("5.3", "POST /api/analyze", bool(r2.get("ok")),
            f"{h.get('status')} / {h.get('total_tokens')} tokens / "
            f"{h.get('report_chars')} 字")

        # 5.4 运营仿真
        r3 = post_json("/api/simulate",
                       {"hours": 24, "agvs": 8, "chargers": 2,
                        "fault_rate": 0.25, "seed": 42})
        ops = r3.get("ops", {})
        add("5.4", "POST /api/simulate", bool(r3.get("ok")),
            f"完成任务 {ops.get('tasks_done')} / "
            f"充电桩 {ops.get('charger_utilization', 0):.1%}" if r3.get("ok") else "")

        # 5.5 页面
        with urllib.request.urlopen(base + "/", timeout=10) as fp:
            html = fp.read().decode("utf-8", errors="ignore")
        has_tabs = ('id="tab-sim"' in html and 'id="tab-analysis"' in html)
        add("5.5", "前端页面（含两个页签）", has_tabs,
            f"{len(html)} 字节")

        return all(x.get("ok") for x in (cfg, r1, r2, r3)) and has_tabs
    except Exception as exc:  # noqa: BLE001
        add("5.x", "Web 接口验证", False, f"{type(exc).__name__}: {exc}")
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()


# --------------------------------------------------------------------------- #
# 6. 3D（可选）
# --------------------------------------------------------------------------- #
def check_3d() -> None:
    print("\n[6/6] 3D 联动（可选）")
    if not port_open(23000):
        add("6.1", "CoppeliaSim 连接", None,
            "未运行 —— 3D 为可选特性，不影响核心链路")
        return

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    try:
        client = RemoteAPIClient(host="localhost", port=23000)
        sim = client.require("sim")
        add("6.1", "ZMQ 握手", True,
            f"版本码 {sim.getInt32Param(sim.intparam_program_version)}")

        found = []
        for n in ("/AGV_01", "/AGV_02", "/AGV_03"):
            try:
                sim.getObject(n)
                found.append(n)
            except Exception:  # noqa: BLE001
                pass
        add("6.2", "场景内 AGV", bool(found),
            ", ".join(found) if found else "无，运行 ./run.sh scene")
    except Exception as exc:  # noqa: BLE001
        add("6.1", "ZMQ 握手", None, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("  AGV 故障智能分析系统 —— 全功能验证")
    print("=" * 74)

    env_ok = check_env()

    core_ok = retrieval_ok = llm_ok = False
    vs = None
    if env_ok:
        try:
            core_ok = check_core()
            from vector_store import VectorStore
            vs = VectorStore()
            retrieval_ok = check_retrieval(vs)
            llm_ok = check_llm(vs)
        except Exception as exc:  # noqa: BLE001
            add("2.x", "核心链路", False, f"{type(exc).__name__}: {exc}")
    else:
        for no, name in (("2", "知识库"), ("3", "召回"), ("4", "LLM")):
            add(no, f"{name}（环境不满足，跳过）", None)

    web_ok = check_web() if env_ok else False
    check_3d()

    # ---- 汇总 ----
    required = [r for r in RESULTS if r[2] in (PASS, FAIL)]
    passed = sum(1 for r in required if r[2] == PASS)
    failed = [r for r in required if r[2] == FAIL]
    optional_warn = [r for r in RESULTS if r[2] == WARN]

    print("\n" + "=" * 74)
    print("  验证汇总")
    print("=" * 74)
    print(f"  必需项通过 : {passed}/{len(required)}")
    if optional_warn:
        print(f"  可选提示   : {len(optional_warn)} 项")
        for no, name, status, detail in optional_warn:
            print(f"     {status} [{no}] {name} {detail}")

    if failed:
        print("\n  未通过项：")
        for no, name, status, detail in failed:
            print(f"     {FAIL} [{no}] {name}  {detail}")
        print("\n" + "=" * 74)
        print("  ❌ 存在未通过项，请按上面提示修复")
        print("=" * 74)
        return 1

    print("\n" + "=" * 74)
    print("  ✅ 核心功能全部通过，可以交付 / 演示")
    print("     主演示：./run.sh web  →  http://127.0.0.1:8000")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

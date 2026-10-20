#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coppelia_bridge.py —— CoppeliaSim ↔ AGV 故障分析链路 桥接器
================================================================================

作用
--------------------------------------------------------------------------------
把 CoppeliaSim 里的 AGV 运动作为事件源，驱动已有的分析链路：

    Python 控制 AGV 沿航线行驶
            ↓  ZMQ
    到达某个区域 → 触发该区域的故障分析
            ↓
    检索 → 规则决策 → LLM（经 Harness 校验）
            ↓
    结果回写场景：区域高亮 + 机器人变色 + 控制台日志

--------------------------------------------------------------------------
为什么由 Python 直接驱动轮子（重要设计决策）
--------------------------------------------------------------------------
CoppeliaSim 的 dr12 模型自带一个 Lua 脚本让它自主避障。实测发现：
一旦通过 ZMQ API 启动仿真，该模型会向场景主脚本注入代码，
导致仿真在 0.15 秒后被主动暂停（getSimulationState() 返回 8）。

验证过程：
    禁用机器人脚本 → 仍暂停
    禁用场景主脚本 → 仿真正常推进

结论：不能依赖模型自带的 Lua 脚本。本桥接器改为：
    1) 启动时禁用场景内所有脚本
    2) 由 Python 通过 setJointTargetVelocity 直接做差速控制
这样行为完全确定、可复现，也不受模型脚本版本变化影响。

--------------------------------------------------------------------------
运动模型：运动学动画
--------------------------------------------------------------------------
本桥接器不使用物理仿真驱动 AGV，而是直接设置其位姿：

    每个控制周期朝目标航点推进 speed × dt，并把朝向对齐到移动方向

为什么不用物理驱动：
    实测本机环境中，CoppeliaSim 的仿真时间推进极不稳定
    （同样 2 秒等待，有时推进 0.0s，有时 1.7s），
    叠加模型脚本会主动暂停仿真，物理驱动无法可靠工作。

    运动学方式的代价是轮子不转、没有碰撞物理；
    但演示要表达的是"AGV 在区域间移动并触发分析"，这一点完全满足，
    且行为百分之百确定、可复现。

--------------------------------------------------------------------------
运行
--------------------------------------------------------------------------------
    ./run.sh bridge                  # 无限运行，Ctrl+C 退出
    ./run.sh bridge --duration 90    # 跑 90 秒后自动退出（录屏用）
    ./run.sh bridge --robot "/dr12[1]"
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 依赖
# --------------------------------------------------------------------------- #
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[FATAL] 缺少 CoppeliaSim Python 客户端。")
    print("        正确运行方式：./run.sh bridge")
    print("        或：./venv/bin/python coppelia_bridge.py")
    raise SystemExit(1)

try:
    import web_demo          # 复用分析链路，保证阈值/重试/降级/指标一致
except ImportError as exc:
    print(f"[FATAL] 无法导入 web_demo：{exc}")
    print("        请确认在 agv_embodied_project 目录下运行（用 ./run.sh bridge）。")
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
ZMQ_HOST = "localhost"
ZMQ_PORT = 23000

DEFAULT_ROBOT = "/dr12[0]"
DEFAULT_SCENE = os.path.join("scenes", "agv_demo.ttt")

POLL_HZ = 20                 # 控制/轮询频率（运动学模式下越高越平滑）
DEFAULT_COOLDOWN = 3.0       # 两次分析之间的最小间隔
DWELL_SECONDS = 4.0          # 到站停留时长：让观众看清结果，也让节奏不至于太赶

# 运动学移动参数
LINEAR_SPEED = 0.55          # m/s（墙钟时间；受轮询频率限制）
WAYPOINT_TOLERANCE = 0.10    # m：距离小于此值视为到达
WAYPOINT_TIMEOUT = 30.0      # s：单个航点最长行驶时间，防卡死。
                             #    达到即原地触发分析，不再等待

# 区域可视化
ZONE_HALF = 2.45
ZONE_LINE_Z = 0.015
ZONE_LINE_COLOR = [0.35, 0.42, 0.55]
HIGHLIGHT_COLORS = {
    "idle": [0.45, 0.50, 0.60],
    "高风险": [1.00, 0.25, 0.25],
    "中风险": [1.00, 0.75, 0.20],
    "低风险": [0.25, 0.85, 0.40],
}
RISK_COLORS = {
    "高风险": [1.00, 0.20, 0.20],
    "中风险": [1.00, 0.75, 0.20],
    "低风险": [0.20, 0.85, 0.35],
}
IDLE_COLOR = [0.70, 0.70, 0.70]

# 四个区域与各自的航点（区域内一点，离边界足够远，避免抖动）
ZONES: List[Dict[str, Any]] = [
    {"area_id": "region_RAN", "name": "装载区", "waypoint": (1.55, 1.55)},
    {"area_id": "region_RAZ", "name": "缓存区", "waypoint": (-1.55, 1.55)},
    {"area_id": "region_RDA", "name": "充电区", "waypoint": (-1.55, -1.55)},
    {"area_id": "region_RAB", "name": "避让区", "waypoint": (1.55, -1.55)},
]

ZONE_CORNER = {
    "region_RAN": (0, 0),
    "region_RAZ": (-1, 0),
    "region_RDA": (-1, -1),
    "region_RAB": (0, -1),
}


# --------------------------------------------------------------------------- #
# 场景准备
# --------------------------------------------------------------------------- #
def make_color(rgb: Any) -> List[float]:
    """把各种颜色表示统一成 RGBA 四元组。

    CoppeliaSim 的 getShapeColor 返回 (flag, [r,g,b]) 形式，
    直接回传会报 "incorrect table size" 或把标志位当成红色分量。
    """
    if isinstance(rgb, (list, tuple)) and len(rgb) == 2 \
            and isinstance(rgb[1], (list, tuple)):
        rgb = rgb[1]
    if rgb is None:
        vals = [0.7, 0.7, 0.7]
    elif isinstance(rgb, (list, tuple)):
        vals = [float(v) for v in rgb]
    else:
        vals = [float(rgb)] * 3
    while len(vals) < 4:
        vals.append(1.0)
    return vals[:4]


def disable_all_scripts(sim: Any) -> int:
    """禁用场景内全部脚本。

    必须在启动仿真前调用：模型自带的 Lua 脚本会主动暂停仿真，
    导致仿真时间停在 0.15 秒附近（状态变为 8 = paused）。
    """
    count = 0
    for h in sim.getObjectsInTree(sim.handle_scene):
        try:
            if sim.getObjectType(h) in (16, 17):     # child script / customization
                sim.setScriptAttribute(h, sim.scriptattribute_enabled, False)
                count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


def find_all_robots(sim: Any, preferred: str) -> List[Dict[str, Any]]:
    """扫描场景中所有二轮差速机器人，返回描述列表。

    每项包含：root / alias / left / right / base_quat
    base_quat 用于保持模型原始姿态（见 move_kinematic 的说明）。
    """
    robots: List[Dict[str, Any]] = []
    seen = set()

    for h in sim.getObjectsInTree(sim.handle_scene):
        try:
            kids = sim.getObjectsInTree(h)
            joints = [k for k in kids if sim.getObjectType(k) == 1]
            if len(joints) != 2:
                continue
            left = right = None
            for k in joints:
                a = sim.getObjectAlias(k, 1).lower()
                if "left" in a:
                    left = k
                elif "right" in a:
                    right = k
            if left is None or right is None:
                continue
            if h in seen:
                continue
            seen.add(h)
            robots.append({
                "root": h,
                "alias": sim.getObjectAlias(h, 1),
                "left": left,
                "right": right,
                "base_quat": tuple(sim.getObjectQuaternion(h, -1)),
                "wp_index": 0,
                "dwell_until": 0.0,
            })
        except Exception:  # noqa: BLE001
            continue

    if not robots:
        raise RuntimeError(
            "场景中找不到二轮差速机器人（需要 2 个含 left/right 的旋转关节）。\n"
            "          可运行 ./run.sh scene 重新生成演示场景。")

    # 把用户指定的那台排到最前（它负责触发分析）
    robots.sort(key=lambda r: (r["alias"] != preferred, r["alias"]))
    return robots


def setup_zone_overlay(sim: Any) -> List[Any]:
    """在地面画出四个区域的边界与彩色高亮框。"""
    S, Z = ZONE_HALF, ZONE_LINE_Z
    handles: List[Any] = []

    static = sim.addDrawingObject(
        sim.drawing_lines, 2, 0.0, -1, 200, ZONE_LINE_COLOR)
    for seg in (
        [-S, -S, Z,  S, -S, Z], [S, -S, Z, S, S, Z],
        [S, S, Z, -S, S, Z], [-S, S, Z, -S, -S, Z],
        [-S, 0, Z,  S, 0, Z], [0, -S, Z, 0, S, Z],
    ):
        sim.addDrawingObjectItem(static, seg)
    handles.append(static)

    for key in ("idle", "高风险", "中风险", "低风险"):
        handles.append(sim.addDrawingObject(
            sim.drawing_lines, 5, 0.0, -1, 20, HIGHLIGHT_COLORS[key]))
    return handles


def draw_zone_highlight(sim: Any, handles: List[Any], area_id: str,
                        color_key: str = "idle") -> None:
    """高亮指定区域（清空其它颜色的框）。"""
    if not handles or len(handles) < 5 or not area_id:
        return
    order = ["idle", "高风险", "中风险", "低风险"]
    try:
        for h in handles[1:]:
            sim.addDrawingObjectItem(h, None)
        target = handles[1 + (order.index(color_key) if color_key in order else 0)]
        S, Z = ZONE_HALF, ZONE_LINE_Z
        cx, cy = ZONE_CORNER.get(area_id, (0, 0))
        x0, x1 = cx * S, (cx + 1) * S
        y0, y1 = cy * S, (cy + 1) * S
        for seg in ([x0, y0, Z, x1, y0, Z], [x1, y0, Z, x1, y1, Z],
                    [x1, y1, Z, x0, y1, Z], [x0, y1, Z, x0, y0, Z]):
            sim.addDrawingObjectItem(target, seg)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 差速驱动
# --------------------------------------------------------------------------- #
def wrap_angle(a: float) -> float:
    """把角度归一化到 (-π, π]。"""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def quat_mul(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]
             ) -> Tuple[float, float, float, float]:
    """四元数乘法 a*b（Hamilton 约定，xyzw 顺序）。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_from_z(angle: float) -> Tuple[float, float, float, float]:
    """绕世界 Z 轴旋转 angle 弧度的四元数。"""
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def move_kinematic(sim: Any, robot: Dict[str, Any],
                   target: Tuple[float, float], speed: float, dt: float,
                   yaw_offset: float) -> float:
    """把 AGV 朝目标点推进一小步，返回剩余距离。

    关于朝向（关键）：
        dr12 模型的原始姿态是 (0, -1.57, 0)，即绕 Y 轴 -90° 才正立。
        如果直接把朝向设成 (0, 0, yaw)，机器人会躺倒 —— 这正是
        "小车倒立"问题的原因。

        正确做法：保留模型原始姿态，再在其上叠加绕世界 Z 轴的偏航：
            q_new = q_z(yaw) * q_base
        用四元数相乘实现，既能转向又不会破坏"正立"。

    为什么不用物理驱动：
        本机环境下仿真时间推进不稳定（同样 2 秒有时只推进 0.0s），
        且模型脚本会主动暂停仿真，物理驱动不可靠。
    """
    root = robot["root"]
    pos = sim.getObjectPosition(root, -1)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)

    if dist < 1e-6:
        return 0.0

    yaw = math.atan2(dy, dx) + yaw_offset
    try:
        q = quat_mul(quat_from_z(yaw), robot["base_quat"])
        sim.setObjectQuaternion(root, -1, list(q))
    except Exception:  # noqa: BLE001
        pass

    step = min(speed * dt, dist)
    sim.setObjectPosition(root, -1, [x + dx / dist * step,
                                     y + dy / dist * step,
                                     z])
    return dist - step


def stop_wheels(sim: Any, left: Any, right: Any) -> None:
    """停止轮子（失败静默）。"""
    try:
        sim.setJointTargetVelocity(left, 0.0)
        sim.setJointTargetVelocity(right, 0.0)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 分析工作线程
# --------------------------------------------------------------------------- #
_res_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()


def _analysis_worker(job: Dict[str, Any]) -> None:
    """在线程中执行分析链路（不触碰 CoppeliaSim API）。"""
    started = time.perf_counter()
    try:
        result = web_demo.do_analyze(job["query_text"], job["area_id"])
        result["_elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        result["_job"] = job
        _res_queue.put(result)
    except Exception as exc:  # noqa: BLE001
        _res_queue.put({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "_job": job,
        })


def pick_event(area_id: str, cursor: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """按区域轮转挑选一条预置故障事件。"""
    candidates = [e for e in web_demo.PRESET_EVENTS if e.get("area_id") == area_id]
    if not candidates:
        return None
    idx = cursor.get(area_id, 0) % len(candidates)
    cursor[area_id] = idx + 1
    return candidates[idx]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CoppeliaSim ↔ AGV 故障分析链路桥接器")
    parser.add_argument("--robot", default=DEFAULT_ROBOT,
                        help=f"机器人对象路径（默认 {DEFAULT_ROBOT}）")
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        help=f"无机器人时自动加载的场景（默认 {DEFAULT_SCENE}）")
    parser.add_argument("--no-auto-scene", action="store_true",
                        help="禁用自动加载场景")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行时长（秒）；0 表示一直运行")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"两次分析最小间隔（默认 {DEFAULT_COOLDOWN}s）")
    parser.add_argument("--host", default=ZMQ_HOST)
    parser.add_argument("--port", type=int, default=ZMQ_PORT)
    parser.add_argument("--yaw-offset", type=float, default=math.pi,
                        help="车头朝向补偿（弧度）。dr12 模型 forward 轴与世界 +x 相反，"
                             "默认 pi；若发现车头朝反方向，改成 0")
    args = parser.parse_args()

    print("=" * 78)
    print("  CoppeliaSim ↔ AGV 故障分析链路 桥接器")
    print("=" * 78)

    # ---- 1) 分析链路 ----
    print("[1/5] 初始化分析链路 ...")
    web_demo.init_runtime()

    # ---- 2) 连接 ----
    print(f"\n[2/5] 连接 CoppeliaSim {args.host}:{args.port} ...")
    try:
        client = RemoteAPIClient(host=args.host, port=args.port)
        try:
            import zmq
            client.socket.setsockopt(zmq.RCVTIMEO, 15000)
            client.socket.setsockopt(zmq.SNDTIMEO, 15000)
            client.socket.setsockopt(zmq.LINGER, 0)
        except Exception:  # noqa: BLE001
            pass
        sim = client.require("sim")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] 连接失败：{type(exc).__name__}: {exc}")
        print("          请确认 CoppeliaSim 已启动。")
        return 1
    print("  ✅ 已连接")

    # ---- 3) 确保仿真停止 ----
    try:
        state = sim.getSimulationState()
        if state != 0:
            print(f"  ⚠️  仿真正在运行（{state}），先停止 ...")
            sim.stopSimulation()
            for _ in range(40):
                if sim.getSimulationState() == 0:
                    break
                time.sleep(0.25)
            if sim.getSimulationState() != 0:
                print("  [FATAL] 无法停止仿真，请在界面点停止按钮后重试。")
                return 1
            print("  ✅ 已停止")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] 无法确认仿真状态：{type(exc).__name__}: {exc}")
        return 1

    # ---- 4) 定位机器人（必要时自动加载场景）----
    try:
        robots = find_all_robots(sim, args.robot)
    except RuntimeError as first_exc:
        scene_path = os.path.abspath(args.scene)
        if args.no_auto_scene or not os.path.isfile(scene_path):
            print(f"  [FATAL] {first_exc}")
            return 1
        print(f"  ℹ️  当前场景没有可用机器人，自动加载：{scene_path}")
        try:
            sim.loadScene(scene_path)
            time.sleep(3.0)
            robots = find_all_robots(sim, args.robot)
            print("  ✅ 场景已加载")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FATAL] 自动加载场景失败：{type(exc).__name__}: {exc}")
            return 1

    primary = robots[0]
    print(f"  ✅ 场景内 {len(robots)} 台 AGV：")
    for r in robots:
        mark = "  ← 负责上报故障" if r is primary else ""
        print(f"     {r['alias']}{mark}")

    try:
        body = sim.getObject(primary["alias"] + "/body")
    except Exception:  # noqa: BLE001
        body = primary["root"]
    try:
        original_color = sim.getShapeColor(body, None, sim.colorcomponent_ambient_diffuse)
    except Exception:  # noqa: BLE001
        original_color = [0.7, 0.7, 0.7]

    # ---- 5) 禁用脚本 + 绘图 ----
    print("\n[3/5] 禁用场景脚本（模型自带脚本会暂停仿真）")
    n_scripts = disable_all_scripts(sim)
    print(f"  已禁用 {n_scripts} 个脚本")

    overlay = setup_zone_overlay(sim)
    print("  ✅ 已绘制区域边界")
    print()
    print("     区域与航点（地面 5m × 5m）：")
    print("       ┌──────────────┬──────────────┐")
    print("       │  缓存区 RAZ   │  装载区 RAN   │   y > 0")
    print("       │ (-1.55, 1.55)│ ( 1.55, 1.55)│")
    print("       ├──────────────┼──────────────┤")
    print("       │  充电区 RDA   │  避让区 RAB   │   y < 0")
    print("       │ (-1.55,-1.55)│ ( 1.55,-1.55)│")
    print("       └──────────────┴──────────────┘")

    print("\n[4/5] 准备就绪（运动学模式：直接设置位姿，不依赖物理引擎）")

    # 各机器人错开起始航点，避免同时挤在一处
    for i, r in enumerate(robots):
        r["wp_index"] = i % len(ZONES)

    # ---- 主循环 ----
    print("\n[5/5] 开始巡游 ...（Ctrl+C 退出）\n")
    print("  " + "-" * 74)
    print(f"  {'时间':>7}  {'位置':>18}  {'区域':<8}  事件")
    print("  " + "-" * 74)

    started_ts = time.time()
    busy = False
    last_trigger_ts = 0.0
    trigger_count = 0
    cursor: Dict[str, int] = {}
    restore_color_at: Optional[float] = None
    dt = 1.0 / POLL_HZ

    try:
        while True:
            now = time.time()
            elapsed = now - started_ts

            if args.duration > 0 and elapsed >= args.duration:
                print(f"  已达到设定时长 {args.duration:.0f}s，准备退出 ...")
                break

            # ---------- 驱动全部 AGV ----------
            for r in robots:
                if now < r["dwell_until"]:
                    continue                     # 到站停留中
                zone = ZONES[r["wp_index"] % len(ZONES)]
                try:
                    remaining = move_kinematic(
                        sim, r, zone["waypoint"], LINEAR_SPEED, dt,
                        args.yaw_offset)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] 移动失败：{type(exc).__name__}: {exc}")
                    continue

                if remaining < WAYPOINT_TOLERANCE:
                    r["dwell_until"] = now + DWELL_SECONDS   # 到站停留
                    r["wp_index"] += 1

            # ---------- 主 AGV 到站则触发分析 ----------
            zone = ZONES[(primary["wp_index"] - 1) % len(ZONES)]
            on_dwell = now < primary["dwell_until"]
            if on_dwell and (not busy) and \
                    (now - last_trigger_ts) >= args.cooldown and \
                    primary.get("dwell_mark") != primary["wp_index"]:

                primary["dwell_mark"] = primary["wp_index"]
                try:
                    pos = sim.getObjectPosition(primary["root"], -1)
                    x, y = float(pos[0]), float(pos[1])
                except Exception:  # noqa: BLE001
                    x = y = 0.0

                draw_zone_highlight(sim, overlay, zone["area_id"], "idle")
                event = pick_event(zone["area_id"], cursor)
                if event:
                    last_trigger_ts = now
                    busy = True
                    print(f"  {elapsed:6.1f}s  ({x:7.3f},{y:7.3f})  "
                          f"{zone['name']:<8}  → {event['label']}")
                    threading.Thread(
                        target=_analysis_worker,
                        args=({"query_text": event["query_text"],
                               "area_id": zone["area_id"],
                               "label": event["label"],
                               "area_name": zone["name"]},),
                        daemon=True,
                    ).start()

            # ---------- 收取分析结果 ----------
            while True:
                try:
                    res = _res_queue.get_nowait()
                except queue.Empty:
                    break

                busy = False
                trigger_count += 1
                job = res.get("_job", {})

                if res.get("ok"):
                    risk = res["risk_level"]
                    action = res["action"]
                    harness = res.get("harness", {})
                    first_line = (res.get("report") or "").strip().split("\n")[0][:60]

                    try:
                        sim.setShapeColor(
                            body, None, sim.colorcomponent_ambient_diffuse,
                            make_color(RISK_COLORS.get(risk, IDLE_COLOR)))
                    except Exception:  # noqa: BLE001
                        pass
                    draw_zone_highlight(sim, overlay, job.get("area_id", ""), risk)
                    restore_color_at = time.time() + 2.5

                    try:
                        sim.addLog(sim.verbosity_scriptinfos,
                                   f"[AGV故障分析] {job.get('area_name','')} · "
                                   f"{job.get('label','')} | {risk} | {action}")
                    except Exception:  # noqa: BLE001
                        pass

                    print(f"           ⤷ 结论：{risk} / {action}")
                    print(f"           ⤷ 报告：{first_line}...")
                    print(f"           ⤷ 耗时：{res.get('_elapsed_ms', 0):.0f} ms | "
                          f"tokens={harness.get('total_tokens', 0)} | "
                          f"status={harness.get('status')}")
                    print("  " + "-" * 74)
                else:
                    print(f"           ⤷ 分析失败：{res.get('error')}")
                    print("  " + "-" * 74)

            # ---------- 颜色恢复 ----------
            if restore_color_at is not None and now >= restore_color_at:
                try:
                    sim.setShapeColor(body, None,
                                      sim.colorcomponent_ambient_diffuse,
                                      make_color(IDLE_COLOR))
                except Exception:  # noqa: BLE001
                    pass
                restore_color_at = None

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\n[exit] 收到 Ctrl+C，正在停止 ...")
    finally:
        for step, fn in (
            ("恢复颜色", lambda: sim.setShapeColor(
                body, None, sim.colorcomponent_ambient_diffuse,
                make_color(original_color))),
            ("清除区域绘制", lambda: [sim.removeDrawingObject(h)
                                      for h in overlay if h is not None]),
        ):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {step}失败：{type(exc).__name__}: {exc}")

    print(f"\n[summary] 共触发 {trigger_count} 次分析")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

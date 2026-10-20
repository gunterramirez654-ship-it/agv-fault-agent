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
import re
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
LINEAR_SPEED = 0.55          # m/s（墙钟时间）
ZONE_HIT_DIST = 0.38         # m：AGV 距区域中心多近算"进入该区域"

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

# 环线路点：按矩形顺序排列，相邻两点只差一个坐标轴 →
# 路径全部沿边线（x=±1.55 或 y=±1.55），不穿越场地中央。
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
    base_quat 在启动时读取，仅用于记录原始姿态（自建车体不需要补偿）。
    """
    robots: List[Dict[str, Any]] = []
    seen = set()

    for h in sim.getObjectsInTree(sim.handle_scene):
        try:
            alias = sim.getObjectAlias(h, 1)
            short = alias.lstrip("/")
            # 只认 AGV_01 / AGV_02 这种根节点名字。
            # 注意不能用 startswith 判断：子部件的别名是 AGV_01_body、
            # AGV_01_nose 这类，startswith 会把它们也当成车去驱动，
            # 结果整车被拆散着乱动。
            if not re.fullmatch(r"AGV_\d+", short):
                continue
            # 根节点必须是 shape（车身）或 dummy；子部件名字带下划线，已被正则排除
            try:
                if sim.getObjectType(h) not in (0, 2):
                    continue
            except Exception:  # noqa: BLE001
                continue
            if h in seen:
                continue
            seen.add(h)
            robots.append({
                "root": h,
                "alias": alias,
                "base_quat": tuple(sim.getObjectQuaternion(h, -1)),
                "dwell_until": 0.0,
            })
        except Exception:  # noqa: BLE001
            continue

    if not robots:
        raise RuntimeError(
            "场景中找不到 AGV（对象名需以 AGV_ 开头）。\n"
            "          可运行 ./run.sh scene 重新生成演示场景。")

    # 把用户指定的那台排到最前（它负责触发分析）
    robots.sort(key=lambda r: (r["alias"] != preferred, r["alias"]))
    return robots


def set_camera_lookat(sim: Any, cam: Any,
                      eye: Tuple[float, float, float],
                      target: Tuple[float, float, float]) -> None:
    """把相机放到 eye，看向 target。

    为什么需要：场景被程序化重建后，默认相机可能贴着地面，
    视野里只剩地板，物体虽然存在却看不到 —— 表现为"3D 视口空白"。
    每次启动时重新摆一次相机，保证全景可见。

    实现：构造 look-at 旋转矩阵（相机沿 +Z 看），再转成四元数。
    """
    ex, ey, ez = eye
    tx, ty, tz = target

    # 视线方向（相机 +Z 轴）
    zx, zy, zz = tx - ex, ty - ey, tz - ez
    norm = math.sqrt(zx * zx + zy * zy + zz * zz) or 1.0
    zx, zy, zz = zx / norm, zy / norm, zz / norm

    # 世界向上 (0,0,1)；若视线几乎垂直，改用 (0,1,0) 避免退化
    if abs(zz) > 0.999:
        upx, upy, upz = 0.0, 1.0, 0.0
    else:
        upx, upy, upz = 0.0, 0.0, 1.0

    # x = up × z
    xx = upy * zz - upz * zy
    xy = upz * zx - upx * zz
    xz = upx * zy - upy * zx
    n = math.sqrt(xx * xx + xy * xy + xz * xz) or 1.0
    xx, xy, xz = xx / n, xy / n, xz / n

    # y = z × x
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx

    # 旋转矩阵 → 四元数
    m00, m01, m02 = xx, yx, zx
    m10, m11, m12 = xy, yy, zy
    m20, m21, m22 = xz, yz, zz
    trace = m00 + m11 + m22
    if trace > 0:
        s2 = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s2
        qx = (m21 - m12) / s2
        qy = (m02 - m20) / s2
        qz = (m10 - m01) / s2
    elif m00 > m11 and m00 > m22:
        s2 = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s2
        qx = 0.25 * s2
        qy = (m01 + m10) / s2
        qz = (m02 + m20) / s2
    elif m11 > m22:
        s2 = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s2
        qx = (m01 + m10) / s2
        qy = 0.25 * s2
        qz = (m12 + m21) / s2
    else:
        s2 = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s2
        qx = (m02 + m20) / s2
        qy = (m12 + m21) / s2
        qz = 0.25 * s2

    try:
        sim.setObjectPosition(cam, -1, [ex, ey, ez])
        sim.setObjectQuaternion(cam, -1, [qx, qy, qz, qw])
    except Exception:  # noqa: BLE001
        pass


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


def catmull_rom_closed(points: List[Tuple[float, float]],
                       samples_per_seg: int = 60) -> List[Tuple[float, float]]:
    """用 Catmull-Rom 样条把控制点连成闭合平滑曲线。

    为什么用样条：
        直接让 AGV 在航点之间走直线，转弯是"折线"，看起来一顿一顿的。
        Catmull-Rom 保证曲线穿过每个控制点，且一二阶导连续 ——
        航点处自然圆滑过渡，视觉上像车在打方向盘。

    参数：
        points          : 控制点（首尾自动闭合）
        samples_per_seg : 每段采样数，越大越平滑

    返回：
        密集的点序列（闭合，首尾不重复）
    """
    n = len(points)
    if n < 3:
        return list(points)

    out: List[Tuple[float, float]] = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) +
                       (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                       (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    return out


def build_path(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """把控制点展开成带弧长索引的样条路径。"""
    dense = catmull_rom_closed(points, samples_per_seg=80)
    # 累计弧长
    cum = [0.0]
    for i in range(1, len(dense)):
        dx = dense[i][0] - dense[i - 1][0]
        dy = dense[i][1] - dense[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    # 闭合段
    dx = dense[0][0] - dense[-1][0]
    dy = dense[0][1] - dense[-1][1]
    total = cum[-1] + math.hypot(dx, dy)
    return {"pts": dense, "cum": cum, "total": max(total, 1e-6)}


def sample_path(path: Dict[str, Any], s_arc: float
                ) -> Tuple[float, float, float]:
    """按弧长取点，返回 (x, y, 切线朝向)。"""
    pts, cum, total = path["pts"], path["cum"], path["total"]
    s_arc %= total

    # 二分查找所在段
    lo, hi = 0, len(cum) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if cum[mid] <= s_arc:
            lo = mid
        else:
            hi = mid
    i = lo
    j = (i + 1) % len(pts)

    seg_len = (cum[j] - cum[i]) if j > i else (total - cum[i] + cum[j])
    t = (s_arc - cum[i]) / seg_len if seg_len > 1e-9 else 0.0
    t = max(0.0, min(1.0, t))

    x = pts[i][0] + (pts[j][0] - pts[i][0]) * t
    y = pts[i][1] + (pts[j][1] - pts[i][1]) * t
    heading = math.atan2(pts[j][1] - pts[i][1], pts[j][0] - pts[i][0])
    return x, y, heading


def advance_along_path(sim: Any, robot: Dict[str, Any], dt: float,
                       speed: float) -> None:
    """沿样条路径推进一小段，并把车头对齐到路径切线。"""
    root = robot["root"]
    s_arc = robot["arc"] + speed * dt
    robot["arc"] = s_arc % robot["path"]["total"]

    x, y, heading = sample_path(robot["path"], robot["arc"])

    # 朝向做一阶平滑：直接取切线会有采样抖动，这里低通一下
    prev = robot.get("yaw", heading)
    diff = wrap_angle(heading - prev)
    yaw = wrap_angle(prev + diff * 0.6)
    robot["yaw"] = yaw

    try:
        pos = sim.getObjectPosition(root, -1)
        sim.setObjectPosition(root, -1, [x, y, float(pos[2])])
        sim.setObjectOrientation(root, -1, [0.0, 0.0, yaw])
    except Exception:  # noqa: BLE001
        pass


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
    parser.add_argument("--yaw-offset", type=float, default=0.0,
                        help="车头朝向补偿（弧度）。自建车体车头朝局部 +X，默认 0；"
                             "若发现车头方向反了，可设为 3.14159")
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
        # 不要给 REQ socket 设 RCVTIMEO：
        # ZMQ 的 REQ 必须严格「发一次、收一次」。一旦某次 recv 超时，
        # 那条未接收的回复会把 socket 留在非法状态，之后每次 send 都会报
        # "Operation cannot be accomplished in current state"，整个连接报废。
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

    # ---- 自动重连 ----
    # 本环境的 ZMQ 连接偶尔会进入坏状态，REQ socket 报
    # "Operation cannot be accomplished in current state"。
    # 对象句柄由服务端维护，重连后依旧有效，因此重建连接即可继续。
    def reconnect() -> bool:
        nonlocal sim, client
        try:
            try:
                client.socket.close(0)
            except Exception:  # noqa: BLE001
                pass
            client = RemoteAPIClient(host=args.host, port=args.port)
            sim = client.require("sim")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 重连失败：{type(exc).__name__}: {exc}")
            return False

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

    # 强制可见：把 AGV 及其部件设到默认可见层，防止被隐藏后"看不见"
    try:
        forced = 0
        for r in robots:
            for h in [r["root"]] + sim.getObjectsInTree(r["root"]):
                try:
                    sim.setObjectInt32Param(
                        h, sim.objintparam_visibility_layer, 1)
                    forced += 1
                except Exception:  # noqa: BLE001
                    pass
        print(f"  ✅ 已强制 {forced} 个对象的可见层为 1")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 设置可见层失败：{type(exc).__name__}: {exc}")

    # 摆正相机：保证能看到整个场地（否则可能只看到地板）
    try:
        cam = sim.getObject("/DefaultCamera")
        set_camera_lookat(sim, cam, (4.2, -4.2, 4.6), (0.0, 0.0, 0.0))
        print("  ✅ 已重置相机视角（俯视全场）")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 重置相机失败：{type(exc).__name__}: {exc}")
    print()
    print("     区域与航点（地面 5m × 5m）：")
    print("       ┌──────────────┬──────────────┐")
    print("       │  缓存区 RAZ   │  装载区 RAN   │   y > 0")
    print("       │ (-1.55, 1.55)│ ( 1.55, 1.55)│")
    print("       ├──────────────┼──────────────┤")
    print("       │  充电区 RDA   │  避让区 RAB   │   y < 0")
    print("       │ (-1.55,-1.55)│ ( 1.55,-1.55)│")
    print("       └──────────────┴──────────────┘")

    print("\n[4/5] 启动仿真（让视图持续刷新）")
    # 为什么会启动仿真：CoppeliaSim 在"仿真停止"状态下不一定持续重绘视图，
    # 只靠 setObjectPosition 改位置，画面可能停在初始状态不动。
    # 现在车体是自建图元、没有任何脚本，启动仿真不会像 dr12 那样被脚本暂停。
    try:
        if sim.getSimulationState() != 0:
            sim.stopSimulation()
            time.sleep(0.5)
        sim.startSimulation()
        time.sleep(0.3)
        print(f"  ✅ 仿真已启动（状态 {sim.getSimulationState()}）")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 启动仿真失败：{type(exc).__name__}: {exc}")
        print("         将继续以运动学方式移动物体")

    # 为每台车构建样条路径；错开起始弧长，三台车分布在不同路段
    control_points = [(z["waypoint"][0], z["waypoint"][1]) for z in ZONES]
    shared_path = build_path(control_points)
    print(f"  路径：{len(shared_path['pts'])} 个采样点，"
          f"闭环弧长 {shared_path['total']:.2f} m")

    for i, r in enumerate(robots):
        r["path"] = shared_path
        r["arc"] = shared_path["total"] * (i / max(len(robots), 1))
        try:
            ori = sim.getObjectOrientation(r["root"], -1)
            r["yaw"] = float(ori[2])
        except Exception:  # noqa: BLE001
            r["yaw"] = 0.0
        r["last_zone"] = None

    # ---- 主循环 ----
    print("\n[5/5] 开始巡游 ...（Ctrl+C 退出）\n")
    print("  " + "-" * 74)
    print(f"  {'时间':>7}  {'位置':>18}  {'区域':<8}  事件")
    print("  " + "-" * 74)

    started_ts = time.time()
    busy = False
    pending_zone: Optional[Dict[str, Any]] = None   # 分析进行中时的待触发区域
    last_trigger_ts = 0.0
    trigger_count = 0
    cursor: Dict[str, int] = {}
    restore_color_at: Optional[float] = None
    move_errors = 0
    dt = 1.0 / POLL_HZ

    try:
        while True:
            now = time.time()
            elapsed = now - started_ts

            if args.duration > 0 and elapsed >= args.duration:
                print(f"  已达到设定时长 {args.duration:.0f}s，准备退出 ...")
                break

            # ---------- 驱动全部 AGV（沿样条路径）----------
            hit_zone: Optional[Dict[str, Any]] = None
            for r in robots:
                if now < r["dwell_until"]:
                    continue                     # 到站停留中

                try:
                    advance_along_path(sim, r, dt, LINEAR_SPEED)
                    pos = sim.getObjectPosition(r["root"], -1)
                except Exception as exc:  # noqa: BLE001
                    move_errors += 1
                    if move_errors <= 3:
                        print(f"  [warn] 移动失败：{type(exc).__name__}: {exc}")
                    if "current state" in str(exc) or "ZMQ" in type(exc).__name__:
                        if reconnect():
                            print("  [info] 已重连 CoppeliaSim，继续")
                            move_errors = 0
                    time.sleep(0.2)
                    continue

                # 判定是否靠近某个区域中心
                px, py = float(pos[0]), float(pos[1])
                for z in ZONES:
                    if math.hypot(z["waypoint"][0] - px,
                                  z["waypoint"][1] - py) < ZONE_HIT_DIST:
                        if r["last_zone"] != z["area_id"]:
                            r["last_zone"] = z["area_id"]
                            if r is primary:
                                hit_zone = {"area_id": z["area_id"],
                                            "name": z["name"],
                                            "x": px, "y": py}
                                r["dwell_until"] = now + DWELL_SECONDS
                        break
                else:
                    # 离开所有区域 → 允许下次再触发
                    r["last_zone"] = None

            # 分析还没结束时先记下，等空闲了再触发
            if hit_zone is not None:
                pending_zone = hit_zone

            # ---------- 主 AGV 到站 → 触发该区域的分析 ----------
            if pending_zone and (not busy) and \
                    (now - last_trigger_ts) >= args.cooldown:
                draw_zone_highlight(sim, overlay, pending_zone["area_id"], "idle")
                event = pick_event(pending_zone["area_id"], cursor)
                if event:
                    last_trigger_ts = now
                    busy = True
                    print(f"  {elapsed:6.1f}s  ({pending_zone['x']:7.3f},"
                          f"{pending_zone['y']:7.3f})  "
                          f"{pending_zone['name']:<8}  → {event['label']}")
                    threading.Thread(
                        target=_analysis_worker,
                        args=({"query_text": event["query_text"],
                               "area_id": pending_zone["area_id"],
                               "label": event["label"],
                               "area_name": pending_zone["name"]},),
                        daemon=True,
                    ).start()
                    pending_zone = None

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
                    try:
                        sim.getSimulationState()      # 探活
                    except Exception:  # noqa: BLE001
                        reconnect()
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
            ("停止仿真", lambda: sim.stopSimulation()),
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

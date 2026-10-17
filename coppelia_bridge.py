#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coppelia_bridge.py —— CoppeliaSim ↔ AGV 故障分析链路 桥接器
================================================================================

作用
--------------------------------------------------------------------------------
把 CoppeliaSim 里 AGV 的**真实运动**作为事件源，驱动已有的分析链路：

    CoppeliaSim 中 AGV 移动
            ↓  ZMQ（10Hz 轮询位置）
    检测到「AGV 进入某区域」
            ↓
    按该区域选一条故障事件
            ↓
    走完整分析链路：向量检索 → 规则决策 → LLM（经 Harness）
            ↓
    结果回写 CoppeliaSim：机器人变色 + 控制台日志

**注意**：故障事件是按区域规则触发的，不是随机乱造；AGV 的运动来自物理仿真。
两个前提严格分离，不做"假数据"。

设计要点
--------------------------------------------------------------------------------
1) 【只读不接管】本脚本只读取 AGV 位置与区域，不控制机器人运动。
   dr12 内置脚本会自主移动，正好用来演示"AGV 在厂区里跑"。

2) 【线程安全】CoppeliaSim 的 ZMQ 客户端不是线程安全的，因此：
   · 主线程：所有 sim.* 调用（读位置、写颜色、写日志）
   · 工作线程：跑分析链路（纯 Python，不碰 sim）
   · 两者通过 queue 通信，避免并发调用 sim

3) 【复用现有链路】直接 import web_demo 的 do_analyze，保证
   阈值、重试、降级、指标与 UI/命令行完全一致，不重复实现。

运行前置
--------------------------------------------------------------------------------
    · CoppeliaSim 已启动，并加载了含移动机器人的场景
    · 机器人对象名默认 /dr12[0]，可用 --robot 指定
    · CoppeliaSim 的 ZMQ Remote API 在监听 23000 端口

运行
--------------------------------------------------------------------------------
    cd /Users/bobo/Desktop/agv_embodied_project
    ./venv/bin/python coppelia_bridge.py

    # 指定其它机器人 / 调整触发间隔
    ./venv/bin/python coppelia_bridge.py --robot "/dr12[1]" --cooldown 8
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 依赖检查
# --------------------------------------------------------------------------- #
def _fail_import(what: str, detail: str = "") -> None:
    """统一处理导入失败，并按实际原因给出正确指引。

    常见误区：用系统 python 而不是 venv 里的 python。
    这时依赖是"装在 venv 里"的，提示去 pip install 只会让人越修越乱，
    因此这里先判断解释器，再决定给哪条建议。
    """
    in_venv = (sys.prefix != sys.base_prefix) or ("venv" in sys.executable.lower())
    print(f"[FATAL] 导入 {what} 失败" + (f"：{detail}" if detail else ""))
    print()
    if not in_venv:
        print("  原因：当前用的不是项目的虚拟环境解释器。")
        print(f"        当前解释器：{sys.executable}")
        print("        依赖装在    ：./venv/bin/python")
        print()
        print("  正确运行方式：")
        print("        ./venv/bin/python coppelia_bridge.py")
        print()
        print("  或者先激活虚拟环境：")
        print("        source venv/bin/activate")
        print("        python coppelia_bridge.py")
    else:
        print("  原因：虚拟环境里缺少该依赖。")
        print("  安装方式：")
        print("        ./venv/bin/python -m pip install pyzmq cbor2 \\")
        print('          "/Applications/coppeliaSim.app/Contents/Resources/programming/'
              'zmqRemoteApi/clients/python"')
    print()
    print("  环境自检：./venv/bin/python doctor.py")
    raise SystemExit(1)


try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError as exc:
    _fail_import("CoppeliaSim Python 客户端", str(exc))

try:
    # 复用 Web 演示服务里的分析链路（阈值/重试/降级/指标完全一致）
    import web_demo
except ImportError as exc:
    _fail_import("项目模块 web_demo", str(exc))


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
ZMQ_HOST = "localhost"
ZMQ_PORT = 23000

DEFAULT_ROBOT = "/dr12[0]"
POLL_HZ = 10                 # 位置轮询频率
DEFAULT_COOLDOWN = 6.0       # 两次触发之间的最小间隔（秒）
WARMUP_SECONDS = 2.0         # 启动后等待时间，避免一上来就触发
HYSTERESIS = 0.30            # 边界滞回带（米）：离分界线过近时保持原区域，避免抖动

# 区域可视化（在 3D 场景里画出四个区域，否则观众看不到"区域"在哪）
ZONE_HALF = 2.45            # 地面半宽（米），略小于 2.5 避免压到墙面
ZONE_LINE_Z = 0.015         # 画线离地高度，避免与地面 z-fighting
ZONE_LINE_COLOR = [0.35, 0.42, 0.55]   # 静态边界线颜色（暗蓝灰）
HIGHLIGHT_COLORS = {        # 区域高亮色（按风险）
    "idle": [0.45, 0.50, 0.60],
    "高风险": [1.00, 0.25, 0.25],
    "中风险": [1.00, 0.75, 0.20],
    "低风险": [0.25, 0.85, 0.40],
}

# 各区域在象限内的角标（用于计算高亮框）
ZONE_CORNER = {
    "region_RAN": (0, 0),      # x+, y+
    "region_RAZ": (-1, 0),     # x-, y+
    "region_RDA": (-1, -1),    # x-, y-
    "region_RAB": (0, -1),     # x+, y-
}

# 区域判定：以原点为中心划分四个象限（与场景的物理布局一致）
# 键为 area_id，值为 (x 判断函数, y 判断函数)
ZONE_RULES: List[Tuple[str, str]] = [
    ("region_RAN", "装载区"),   # x >= 0, y >= 0
    ("region_RAZ", "缓存区"),   # x <  0, y >= 0
    ("region_RDA", "充电区"),   # x <  0, y <  0
    ("region_RAB", "避让区"),   # x >= 0, y <  0
]

# 风险等级 → 机器人颜色（RGB，0~1）
RISK_COLORS: Dict[str, List[float]] = {
    "高风险": [1.00, 0.20, 0.20],
    "中风险": [1.00, 0.75, 0.20],
    "低风险": [0.20, 0.85, 0.35],
}
IDLE_COLOR = [0.70, 0.70, 0.70]


def setup_zone_overlay(sim: Any) -> List[Any]:
    """在 3D 场景中画出四个区域的边界，并为每种风险色预建一个高亮框。

    为什么需要：区域判定是基于世界坐标象限的纯数学划分，3D 里本来什么都看不见。
    不画出来的话，观众无法理解"AGV 进入了装载区"这个判断从何而来。

    为什么每种颜色单独建对象：CoppeliaSim 的绘图对象在创建时固定颜色，
    没有可靠的运行时改色接口。与其冒险调用不存在的 API，不如按颜色各建一个，
    每次只往目标对象里画、把其它清空。

    返回：
        绘图对象列表：[静态边界, 高亮_idle, 高亮_high, 高亮_mid, 高亮_low]
    """
    S = ZONE_HALF
    Z = ZONE_LINE_Z
    handles: List[Any] = []

    # ---- 静态边界：外框 + 中轴线 ----
    static = sim.addDrawingObject(
        sim.drawing_lines, 2, 0.0, -1, 200, ZONE_LINE_COLOR)
    for seg in (
        [-S, -S, Z,  S, -S, Z],   # 下边
        [ S, -S, Z,  S,  S, Z],   # 右边
        [ S,  S, Z, -S,  S, Z],   # 上边
        [-S,  S, Z, -S, -S, Z],   # 左边
        [-S,  0, Z,  S,  0, Z],   # 中轴 y=0
        [ 0, -S, Z,  0,  S, Z],   # 中轴 x=0
    ):
        sim.addDrawingObjectItem(static, seg)
    handles.append(static)

    # ---- 按颜色预建高亮框 ----
    for key in ("idle", "高风险", "中风险", "低风险"):
        h = sim.addDrawingObject(
            sim.drawing_lines, 5, 0.0, -1, 20, HIGHLIGHT_COLORS[key])
        handles.append(h)

    return handles


def draw_zone_highlight(sim: Any, handles: List[Any], area_id: str,
                        color_key: str = "idle") -> None:
    """把指定区域用彩色框高亮（清空其它颜色的框）。"""
    if not handles or len(handles) < 5 or not area_id:
        return
    order = ["idle", "高风险", "中风险", "低风险"]
    try:
        # 清空全部高亮框
        for h in handles[1:]:
            sim.addDrawingObjectItem(h, None)

        target = handles[1 + (order.index(color_key)
                              if color_key in order else 0)]

        S = ZONE_HALF
        Z = ZONE_LINE_Z
        cx, cy = ZONE_CORNER.get(area_id, (0, 0))
        x0, x1 = cx * S, (cx + 1) * S
        y0, y1 = cy * S, (cy + 1) * S
        for seg in (
            [x0, y0, Z, x1, y0, Z],
            [x1, y0, Z, x1, y1, Z],
            [x1, y1, Z, x0, y1, Z],
            [x0, y1, Z, x0, y0, Z],
        ):
            sim.addDrawingObjectItem(target, seg)
    except Exception:  # noqa: BLE001  绘图失败不应影响主流程
        pass


def find_robot(sim: Any, preferred: str) -> Tuple[Any, str]:
    """定位要跟踪的机器人对象。

    优先用指定路径；找不到时自动扫描场景里带两个以上关节的对象
    （差分驱动 / 移动机器人的典型特征），避免因为对象改名而跑不起来。

    参数：
        sim       : CoppeliaSim 的 sim 模块
        preferred : 用户指定的对象路径

    返回：
        (对象句柄, 别名)；找不到时抛 RuntimeError
    """
    # ---- 优先用指定路径 ----
    try:
        h = sim.getObject(preferred)
        return h, sim.getObjectAlias(h, 1)
    except Exception:  # noqa: BLE001
        pass

    # ---- 自动扫描 ----
    candidates: List[Tuple[Any, str, int]] = []
    for h in sim.getObjectsInTree(sim.handle_scene):
        try:
            kids = sim.getObjectsInTree(h)
            joints = sum(1 for k in kids if sim.getObjectType(k) == 1)
            if joints < 2:
                continue
            alias = sim.getObjectAlias(h, 1)
            # 排除机械臂：移动机器人通常关节少（2~4 个），机械臂是 6~7 个
            if joints <= 4:
                candidates.append((h, alias, joints))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"场景中找不到可跟踪的移动机器人（指定路径 {preferred!r} 也不存在）。\n"
            f"          可用 --robot 指定，或先运行 ./run.sh scene 生成演示场景。")

    # 命名优先：AGV / dr12 / Pioneer / ePuck 等
    for kw in ("AGV", "agv", "dr12", "Pioneer", "ePuck", "Robot"):
        for h, alias, _ in candidates:
            if kw in alias:
                return h, alias

    h, alias, _ = candidates[0]
    return h, alias


def detect_zone(x: float, y: float, prev: Optional[str] = None) -> Tuple[str, str]:
    """根据坐标判定 AGV 所在区域，返回 (area_id, 中文名)。

    带滞回（hysteresis）：当 AGV 距离分界线小于 HYSTERESIS 时，保持上一次的区域。
    原因：AGV 停在 x=0 或 y=0 附近时，坐标会因物理抖动在分界线两侧来回跳，
         不加滞回会疯狂触发"区域切换"，把分析链路打爆。

    参数：
        x, y : AGV 世界坐标
        prev : 上一次判定的 area_id（可为 None）
    """
    # 落在边界死区内 → 沿用上次区域
    if prev is not None and (abs(x) < HYSTERESIS or abs(y) < HYSTERESIS):
        return prev, AREA_NAMES.get(prev, "未知")

    if x >= 0 and y >= 0:
        return "region_RAN", "装载区"
    if x < 0 and y >= 0:
        return "region_RAZ", "缓存区"
    if x < 0 and y < 0:
        return "region_RDA", "充电区"
    return "region_RAB", "避让区"


# area_id → 中文名（滞回分支用）
AREA_NAMES: Dict[str, str] = {
    "region_RAN": "装载区",
    "region_RAZ": "缓存区",
    "region_RDA": "充电区",
    "region_RAB": "避让区",
}


def pick_event(area_id: str, cursor: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """按区域从预置事件里挑一条，轮转选择，避免每次都触发同一事件。"""
    candidates = [e for e in web_demo.PRESET_EVENTS if e.get("area_id") == area_id]
    if not candidates:
        return None
    idx = cursor.get(area_id, 0) % len(candidates)
    cursor[area_id] = idx + 1
    return candidates[idx]


def make_color(rgb: Any) -> List[float]:
    """把各种颜色表示统一成 CoppeliaSim 需要的四元组（RGBA）。

    CoppeliaSim 的颜色返回值有多种形态，必须统一处理，否则会报
    "incorrect table size" 或把标志位误当成红色分量：

        [0.7, 0.7, 0.7]              纯 RGB
        [0.7, 0.7, 0.7, 1.0]         纯 RGBA
        (1, [1.0, 0.75, 0.2])        getShapeColor 的实际返回：
                                     第 0 位是标志位，第 1 位才是 RGB

    参数：
        rgb : 上述任一形态的颜色，或 None

    返回：
        [r, g, b, a] 四元组，各分量 0~1
    """
    # ---- 形态 3：getShapeColor 返回 (flag, [r,g,b]) ----
    if isinstance(rgb, (list, tuple)) and len(rgb) == 2 \
            and isinstance(rgb[1], (list, tuple)):
        rgb = rgb[1]

    # ---- 归一为列表 ----
    if rgb is None:
        vals: List[float] = [0.7, 0.7, 0.7]
    elif isinstance(rgb, (list, tuple)):
        vals = [float(v) for v in rgb]
    else:
        vals = [float(rgb)] * 3

    # ---- 补足到 4 个分量（alpha 默认 1.0）----
    while len(vals) < 4:
        vals.append(1.0)
    return vals[:4]


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
    except Exception as exc:  # noqa: BLE001  任何异常都回传，不吞掉
        _res_queue.put({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "_job": job,
        })


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CoppeliaSim ↔ AGV 故障分析链路桥接器"
    )
    parser.add_argument("--robot", default=DEFAULT_ROBOT,
                        help=f"机器人对象路径（默认 {DEFAULT_ROBOT}）")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"两次触发的最小间隔秒数（默认 {DEFAULT_COOLDOWN}）")
    parser.add_argument("--host", default=ZMQ_HOST, help="CoppeliaSim 主机")
    parser.add_argument("--port", type=int, default=ZMQ_PORT, help="ZMQ 端口")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行时长（秒）；0 表示一直运行直到 Ctrl+C")
    args = parser.parse_args()

    print("=" * 78)
    print("  CoppeliaSim ↔ AGV 故障分析链路 桥接器")
    print("=" * 78)

    # ---- 1) 初始化分析链路 ----
    print("[1/4] 初始化分析链路 ...")
    web_demo.init_runtime()

    # ---- 2) 连接 CoppeliaSim ----
    print(f"\n[2/4] 连接 CoppeliaSim {args.host}:{args.port} ...")
    try:
        client = RemoteAPIClient(host=args.host, port=args.port)
        sim = client.require("sim")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] 连接失败：{type(exc).__name__}: {exc}")
        print("          请确认 CoppeliaSim 已启动且 ZMQ Remote API 在运行。")
        return 1

    # ---- 3) 定位机器人（自动识别，避免因对象改名而失败）----
    try:
        robot, robot_alias = find_robot(sim, args.robot)
    except RuntimeError as exc:
        print(f"  [FATAL] {exc}")
        return 1

    if robot_alias != args.robot:
        print(f"  ℹ️  指定的 {args.robot!r} 不存在，自动选用 {robot_alias!r}")
    print(f"  ✅ 跟踪对象：{robot_alias}")

    # body 子对象用于变色；找不到就用根对象
    try:
        body = sim.getObject(robot_alias + "/body")
    except Exception:  # noqa: BLE001
        body = robot

    original_color = sim.getShapeColor(body, None, sim.colorcomponent_ambient_diffuse)
    print(f"     当前位置：{sim.getObjectPosition(robot, -1)}")

    # ---- 在场景中画出四个区域（否则观众看不到"区域"在哪）----
    overlay = setup_zone_overlay(sim)
    print("  ✅ 已在场景中绘制区域边界")
    print()
    print("     区域分布（俯视，地面 5m × 5m）：")
    print("       ┌──────────────┬──────────────┐")
    print("       │  缓存区 RAZ   │  装载区 RAN   │   y > 0")
    print("       │  x<0, y>0    │  x>0, y>0    │")
    print("       ├──────────────┼──────────────┤")
    print("       │  充电区 RDA   │  避让区 RAB   │   y < 0")
    print("       │  x<0, y<0    │  x>0, y<0    │")
    print("       └──────────────┴──────────────┘")
    print("            x < 0           x > 0")

    # ---- 4) 启动仿真并进入监听循环 ----
    print(f"\n[3/4] 启动仿真（轮询 {POLL_HZ}Hz，触发冷却 {args.cooldown}s）")
    sim.startSimulation()

    current_zone: Optional[str] = None
    last_trigger_ts = 0.0
    restore_color_at: Optional[float] = None   # 颜色恢复的截止时间（主线程处理）
    started_ts = time.time()
    event_cursor: Dict[str, int] = {}
    trigger_count = 0
    busy = False

    print("\n[4/4] 监听中 ...（Ctrl+C 退出）\n")
    print("  " + "-" * 74)
    print(f"  {'时间':>7}  {'位置':>18}  {'区域':<8}  事件")
    print("  " + "-" * 74)

    try:
        while True:
            now = time.time()
            elapsed = now - started_ts

            # 定长运行：到达设定时长后正常退出（便于录屏与脚本化演示）
            if args.duration > 0 and elapsed >= args.duration:
                print(f"  已达到设定时长 {args.duration:.0f}s，准备退出 ...")
                break

            # ---------- 读取位置（主线程，唯一调用 sim 的地方）----------
            pos = sim.getObjectPosition(robot, -1)
            x, y = float(pos[0]), float(pos[1])
            area_id, area_name = detect_zone(x, y, prev=current_zone)

            # ---------- 区域切换检测 ----------
            zone_changed = area_id != current_zone
            if zone_changed:
                current_zone = area_id

            # ---------- 是否触发分析 ----------
            can_trigger = (
                zone_changed
                and elapsed >= WARMUP_SECONDS
                and not busy
                and (now - last_trigger_ts) >= args.cooldown
            )

            if can_trigger:
                event = pick_event(area_id, event_cursor)
                if event:
                    last_trigger_ts = now
                    busy = True
                    print(f"  {elapsed:6.1f}s  ({x:7.3f},{y:7.3f})  {area_name:<8}  "
                          f"→ {event['label']}")
                    # 分析在子线程执行，主线程继续轮询位置
                    threading.Thread(
                        target=_analysis_worker,
                        args=({"query_text": event["query_text"],
                               "area_id": area_id,
                               "label": event["label"],
                               "area_name": area_name},),
                        daemon=True,
                    ).start()
            elif zone_changed:
                print(f"  {elapsed:6.1f}s  ({x:7.3f},{y:7.3f})  {area_name:<8}  "
                      f"（冷却中，跳过）")

            # 区域切换时在场景中高亮当前区域（灰色待定）
            if zone_changed:
                draw_zone_highlight(sim, overlay, area_id, "idle")

            # ---------- 颜色恢复（仅主线程操作 sim）----------
            if restore_color_at is not None and now >= restore_color_at:
                try:
                    sim.setShapeColor(body, None,
                                      sim.colorcomponent_ambient_diffuse,
                                      make_color(IDLE_COLOR))
                except Exception:  # noqa: BLE001
                    pass
                restore_color_at = None

            # ---------- 收取分析结果并回写 CoppeliaSim ----------
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
                    report_first = (res.get("report") or "").strip().split("\n")[0][:60]

                    # 回写 1：机器人变色
                    color = RISK_COLORS.get(risk, IDLE_COLOR)
                    sim.setShapeColor(body, None,
                                      sim.colorcomponent_ambient_diffuse,
                                      make_color(color))
                    # 回写 2：区域高亮成风险色（观众一眼看到"哪个区出了问题"）
                    draw_zone_highlight(sim, overlay, job.get("area_id", ""), risk)
                    # 回写 2：控制台日志
                    sim.addLog(
                        sim.verbosity_scriptinfos,
                        f"[AGV故障分析] {job.get('area_name','')} · {job.get('label','')} | "
                        f"{risk} | {action} | {report_first}",
                    )
                    print(f"           ⤷ 结论：{risk} / {action}")
                    print(f"           ⤷ 报告：{report_first}...")
                    print(f"           ⤷ 耗时：{res.get('_elapsed_ms', 0):.0f} ms | "
                          f"tokens={harness.get('total_tokens', 0)} | "
                          f"status={harness.get('status')}")
                    print("  " + "-" * 74)

                    # 颜色恢复：记录截止时间，由主循环统一处理
                    # 注意：绝不能在这里起 Timer 线程去调 sim.*，
                    #      ZMQ REQ socket 非线程安全，跨线程调用会破坏连接状态。
                    restore_color_at = time.time() + 2.0
                else:
                    print(f"           ⤷ 分析失败：{res.get('error')}")
                    print("  " + "-" * 74)

            time.sleep(1.0 / POLL_HZ)

    except KeyboardInterrupt:
        print("\n\n[exit] 收到 Ctrl+C，正在停止 ...")
    finally:
        # 退出时逐项保护：任何一步失败都不影响其它清理动作
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

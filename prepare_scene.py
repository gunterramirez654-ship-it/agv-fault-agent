#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_scene.py —— 生成 CoppeliaSim 演示场景
================================================================================

为什么需要这个脚本
--------------------------------------------------------------------------------
CoppeliaSim 自带的场景里没有现成的"多台 AGV + 障碍物"布局，而演示又需要
一个稳定、可复现的场景。手工在界面里拖模型既费时又容易漏。

本脚本通过 ZMQ 远程 API 程序化地搭出场景并保存为 .ttt 文件：

    3 台 dr12 差分驱动机器人（CoppeliaSim 自带模型）
    6 个圆柱障碍物（分布在四个象限，模拟货架/立柱）
    地面、灯光、相机使用场景默认物

生成后可用 CoppeliaSim 直接打开 scenes/agv_demo.ttt，
再运行 ./run.sh bridge 即可演示。

运行
--------------------------------------------------------------------------------
    # 需要 CoppeliaSim 已启动
    ./run.sh scene

    # 或手动
    ./venv/bin/python prepare_scene.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[FATAL] 缺少 CoppeliaSim 客户端，请先执行：")
    print("        ./venv/bin/python -m pip install pyzmq cbor2 \\")
    print('          "~/Applications/coppeliaSim.app/Contents/Resources/'
          'programming/zmqRemoteApi/clients/python"')
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 场景布局
# --------------------------------------------------------------------------- #
# 地面为 5m × 5m，坐标范围 -2.5 ~ +2.5
# 机器人初始位置（避开障碍物，且分散在四个象限）
ROBOT_POSITIONS: List[Tuple[float, float]] = [
    (0.30, -1.20),     # 避让区（x>0, y<0）
    (-1.20, 0.90),     # 缓存区（x<0, y>0）
    (1.40, 1.10),      # 装载区（x>0, y>0）
]

# 障碍物位置与半径（模拟货架立柱）
OBSTACLES: List[Tuple[float, float, float]] = [
    (1.90, 1.90, 0.25),
    (-1.90, 1.90, 0.25),
    (1.90, -1.90, 0.25),
    (-1.90, -1.90, 0.25),
    (0.90, 0.60, 0.20),
    (-0.80, -0.60, 0.20),
]

MODEL_REL_PATH = "Contents/Resources/models/robots/mobile/dr12.ttm"

# 常见安装位置（按优先级）
APP_CANDIDATES = [
    os.path.expanduser("~/Applications/coppeliaSim.app"),
    "/Applications/coppeliaSim.app",
    os.path.expanduser("~/Downloads/coppeliaSim.app"),
]

SCENE_OUT = os.path.join("scenes", "agv_demo.ttt")


def find_model_path() -> str:
    """在常见安装位置中查找 dr12 模型文件。"""
    for app in APP_CANDIDATES:
        p = os.path.join(app, MODEL_REL_PATH)
        if os.path.isfile(p):
            return p
    print("[FATAL] 未找到 dr12 模型文件，已尝试：")
    for app in APP_CANDIDATES:
        print(f"        {os.path.join(app, MODEL_REL_PATH)}")
    print("        可用 --model 手动指定路径")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 CoppeliaSim 演示场景")
    parser.add_argument("--model", default="", help="dr12.ttm 的完整路径（自动查找失败时使用）")
    parser.add_argument("--out", default=SCENE_OUT, help=f"输出场景路径（默认 {SCENE_OUT}）")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--keep-existing", action="store_true",
                        help="保留场景中已有的 dr12 与圆柱（默认会先清理，保证可重复执行）")
    args = parser.parse_args()

    print("=" * 74)
    print("  CoppeliaSim 演示场景生成")
    print("=" * 74)

    model_path = args.model or find_model_path()
    print(f"\n[1/4] 模型文件：{model_path}")

    print(f"[2/4] 连接 CoppeliaSim {args.host}:{args.port} ...")
    try:
        client = RemoteAPIClient(host=args.host, port=args.port)
        sim = client.require("sim")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] 连接失败：{type(exc).__name__}: {exc}")
        print("          请先启动 CoppeliaSim。")
        return 1
    print("  ✅ 已连接")

    # ---- 清理已有对象（保证脚本可重复执行）----
    if not args.keep_existing:
        removed = 0
        for h in sim.getObjectsInTree(sim.handle_scene):
            try:
                alias = sim.getObjectAlias(h, 0)
            except Exception:
                continue
            if alias.startswith(("dr12", "AGV_", "Cylinder")):
                try:
                    sim.removeObjects([h])
                    removed += 1
                except Exception:
                    pass
        if removed:
            print(f"  已清理场景中 {removed} 个旧对象")

    # ---- 加载机器人 ----
    print(f"\n[3/4] 加载 {len(ROBOT_POSITIONS)} 台机器人 ...")
    robots: List[int] = []
    for i, (x, y) in enumerate(ROBOT_POSITIONS, 1):
        try:
            h = sim.loadModel(model_path)
            sim.setObjectPosition(h, -1, [x, y, 0.077])
            # 注意：不要给模型改名。
            # dr12 的内置脚本按原始对象名查找关节，改名后脚本会报错，
            # 进而导致仿真在开始后不久被中止（表现为"机器人不动"）。
            # 保留默认名 dr12，CoppeliaSim 会自动编号为 dr12[0]、dr12[1]…
            robots.append(h)
            alias = sim.getObjectAlias(h, 1)
            print(f"  ✅ {alias}  位置 ({x:+.2f}, {y:+.2f})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ 第 {i} 台加载失败：{type(exc).__name__}: {exc}")

    # ---- 创建障碍物 ----
    print(f"\n[4/4] 创建 {len(OBSTACLES)} 个圆柱障碍物 ...")
    created = 0
    for x, y, r in OBSTACLES:
        try:
            h = sim.createPrimitiveShape(
                sim.primitiveshape_cylinder, [r, r, 0.60], 0)
            sim.setObjectPosition(h, -1, [x, y, 0.30])
            created += 1
        except Exception:  # noqa: BLE001
            pass
    print(f"  ✅ 已创建 {created} 个")

    # ---- 保存场景 ----
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    abs_out = os.path.abspath(args.out)

    print(f"\n[保存] 写入场景文件 ...")
    try:
        sim.saveScene(abs_out)
        size = os.path.getsize(abs_out) / 1024 / 1024 if os.path.isfile(abs_out) else 0
        print(f"  ✅ {abs_out}  ({size:.2f} MB)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 保存失败：{type(exc).__name__}: {exc}")
        return 1

    print("\n" + "=" * 74)
    print("  完成。使用方法：")
    print("=" * 74)
    print(f"  1) 在 CoppeliaSim 中打开：{abs_out}")
    print("     （菜单 File → Open scene，或直接双击）")
    print()
    print("  2) 另开终端运行桥接器：")
    print("     ./run.sh bridge")
    print()
    print("  3) 若跟踪的不是第一台，可指定：")
    print("     ./run.sh bridge --robot \"/AGV_02\"")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

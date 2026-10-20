#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doctor.py —— 环境自检工具
================================================================================

一键检查项目运行所需的全部前置条件，并给出明确的修复建议。
出现任何报错时，先跑这个脚本，比逐条猜要快得多。

用法：
    cd /Users/bobo/Desktop/agv_embodied_project
    ./venv/bin/python doctor.py

检查项分三档：
    [核心] 不通就跑不了命令行链路
    [UI]   不通就跑不了 Web 界面
    [3D]   不通就跑不了 CoppeliaSim 桥接
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from typing import List, Tuple

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results: List[Tuple[str, bool, str]] = []   # (名称, 是否通过, 修复建议)


# 3D 联动是可选特性：它不通过时只警告，不算失败
OPTIONAL_LEVELS = {"3D", "UI"}


def record(level: str, name: str, ok: bool, hint: str = "") -> None:
    """记录一项检查结果。

    核心依赖（向量模型、LLM 配置等）不通过 → 计入失败。
    可选依赖（3D 联动、Web 服务）不通过 → 只提示，不阻断。
    """
    optional = level in OPTIONAL_LEVELS
    if not (optional and not ok):
        results.append((f"[{level}] {name}", ok, hint))

    if ok:
        mark = PASS
    else:
        mark = WARN if optional else FAIL
    print(f"  {mark} [{level}] {name}")
    if not ok and hint:
        prefix = "（可选，不影响核心链路）" if optional else ""
        print(f"        → {prefix}{hint}")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> None:
    """检查 Python 解释器是否为项目 venv。"""
    print("\n【1】Python 解释器")
    exe = sys.executable
    in_venv = "venv" in exe.lower() or sys.prefix != sys.base_prefix
    record("核心", f"解释器：{exe}", in_venv,
           "请使用 ./venv/bin/python 运行，而不是系统 python3\n"
           "        正确写法：./venv/bin/python coppelia_bridge.py")

    print(f"       版本：{sys.version.split()[0]}")


def check_cwd() -> None:
    """检查工作目录是否为项目根目录。"""
    print("\n【2】工作目录")
    cwd = os.getcwd()
    has_kb = os.path.isfile("knowledge_base.json")
    record("核心", f"当前目录：{cwd}", has_kb,
           "请在项目根目录运行：cd /Users/bobo/Desktop/agv_embodied_project")


def check_packages() -> None:
    """检查 Python 依赖。"""
    print("\n【3】Python 依赖包")
    required = [
        ("core", "chromadb", "向量化"),
        ("core", "onnxruntime", "ONNX 推理"),
        ("core", "tokenizers", "分词"),
        ("core", "numpy", "数值计算"),
        ("core", "openai", "LLM 调用"),
        ("3d", "zmq", "CoppeliaSim 通信"),
        ("3d", "cbor2", "CoppeliaSim 序列化"),
        ("3d", "coppeliasim_zmqremoteapi_client", "CoppeliaSim 客户端"),
    ]
    for level, mod, desc in required:
        lv = "核心" if level == "core" else "3D"
        ok = has_module(mod)
        hint = ""
        if not ok:
            if level == "3d":
                hint = ("./venv/bin/python -m pip install pyzmq cbor2 \\\n"
                        '          "/Applications/coppeliaSim.app/Contents/Resources/'
                        'programming/zmqRemoteApi/clients/python"')
            else:
                hint = "./venv/bin/python -m pip install -r requirements.txt"
        record(lv, f"{mod:34} ({desc})", ok, hint)


def check_files() -> None:
    """检查项目数据文件与模型。"""
    print("\n【4】项目文件")
    files = [
        ("knowledge_base.json", "核心", "知识库"),
        ("vector_store.py", "核心", "召回模块"),
        ("bge_embedding.py", "核心", "向量化模块"),
        ("agv_agent.py", "核心", "LLM 模块"),
        ("agent_harness.py", "核心", "可靠性外壳"),
        ("simple_logger.py", "核心", "日志模块"),
        ("test_vector_store.py", "核心", "命令行入口"),
        ("web_demo.py", "UI", "Web 服务"),
        ("templates/index.html", "UI", "界面文件"),
        ("coppelia_bridge.py", "3D", "桥接器"),
    ]
    for path, level, desc in files:
        record(level, f"{path:24} ({desc})", os.path.isfile(path),
               f"文件缺失：{path}")

    # ---- 模型文件 ----
    print("\n【5】BGE 向量模型")
    model_dir = os.getenv("BGE_MODEL_DIR") or "onnx_models/bge-small-zh-v1.5/onnx"
    need = ["model.onnx", "tokenizer.json", "config.json"]
    if not os.path.isdir(model_dir):
        record("核心", f"模型目录 {model_dir}", False,
               "模型未就位。需把 BGE ONNX 模型放到 onnx_models/bge-small-zh-v1.5/onnx/\n"
               "        详见 README 第 8.2 节")
        return
    for f in need:
        p = os.path.join(model_dir, f)
        size = os.path.getsize(p) / 1024 / 1024 if os.path.isfile(p) else 0
        record("核心", f"{f:20} ({size:.1f} MB)" if size else f"{f:20}",
               os.path.isfile(p), f"缺少模型文件：{p}")

    # tokenizer 体积校验：MiniLM 误放会在这里暴露
    tok = os.path.join(model_dir, "tokenizer.json")
    if os.path.isfile(tok):
        mb = os.path.getsize(tok) / 1024 / 1024
        ok = 0.2 < mb < 5.0
        record("核心", f"tokenizer.json 体积校验 ({mb:.2f} MB)", ok,
               "体积异常，可能放错了模型（MiniLM 的 tokenizer 约 0.68MB，"
               "BGE 约 0.42MB）")


def check_env() -> None:
    """检查 .env 配置。"""
    print("\n【6】环境变量 / .env")
    if not os.path.isfile(".env"):
        record("核心", ".env 文件", False,
               "执行：cp .env.example .env，然后填入 DEEPSEEK_API_KEY")
        return
    record("核心", ".env 文件存在", True)

    content = {}
    with open(".env", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                content[k.strip()] = v.strip()

    key = content.get("DEEPSEEK_API_KEY", "")
    ok = bool(key) and key != "sk-placeholder-for-validation-only"
    record("核心", "DEEPSEEK_API_KEY 已配置",
           ok, ".env 里 DEEPSEEK_API_KEY 为空或仍是占位值")

    model = content.get("AGV_AGENT_MODEL", "")
    record("核心", f"AGV_AGENT_MODEL = {model or '未设置'}",
           bool(model),
           "未设置时 agv_agent 会回落到默认模型 deepseek-chat（可能不可用）")


def check_coppelia() -> None:
    """检查 CoppeliaSim 是否在运行并可连接。"""
    print("\n【7】CoppeliaSim 连接")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    reachable = False
    try:
        s.connect(("127.0.0.1", 23000))
        reachable = True
    except Exception:  # noqa: BLE001
        pass
    finally:
        s.close()

    record("3D", "端口 23000 可连接", reachable,
           "CoppeliaSim 未启动，或 ZMQ Remote API 未开启")
    if not reachable:
        return

    # 进程检查（macOS 上 ps 可能被限制，失败不算错）
    try:
        out = subprocess.run(["pgrep", "-fl", "coppeliaSim"],
                             capture_output=True, text=True, timeout=5)
        running = bool(out.stdout.strip())
        record("3D", "CoppeliaSim 进程存在", running, "未检测到进程")
        if running:
            first = out.stdout.strip().split("\n")[0]
            if "AppTranslocation" in first:
                print(f"       {WARN}正在从 AppTranslocation 运行（Gatekeeper 沙箱）")
                print("          建议把 app 拖到 /Applications 再运行，路径更稳定")
    except Exception:  # noqa: BLE001
        print(f"       {WARN}无法检查进程（沙箱限制），跳过")

    # 实际尝试调用 API
    if has_module("coppeliasim_zmqremoteapi_client"):
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            client = RemoteAPIClient(host="localhost", port=23000)
            sim = client.require("sim")
            ver = sim.getInt32Param(sim.intparam_program_version)
            record("3D", f"ZMQ 握手成功（版本码 {ver}）", True)

            # 检查场景里是否有可跟踪的 AGV（兼容自建 AGV_xx 与旧 dr12）
            found = []
            for name in ("/AGV_01", "/AGV_02", "/AGV_03", "/dr12[0]"):
                try:
                    sim.getObject(name)
                    found.append(name)
                except Exception:
                    pass
            if found:
                record("3D", f"场景内有可跟踪的 AGV（{', '.join(found[:3])}）", True)
            else:
                record("3D", "场景内有可跟踪的 AGV", False,
                       "当前场景没有 AGV。运行 ./run.sh scene 生成演示场景，\n"
                       "        或在 CoppeliaSim 里打开 scenes/agv_demo.ttt")
        except Exception as exc:  # noqa: BLE001
            record("3D", "ZMQ 握手", False, f"{type(exc).__name__}: {exc}")


def check_ports() -> None:
    """检查 Web 服务端口。"""
    print("\n【8】Web 服务")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    up = False
    try:
        s.connect(("127.0.0.1", 8000))
        up = True
    except Exception:  # noqa: BLE001
        pass
    finally:
        s.close()
    # 未运行不算错误（可选组件）
    mark = PASS if up else WARN
    print(f"  {mark} [UI] 端口 8000 {'已监听（服务在跑）' if up else '未监听（可选，运行 web_demo.py 启动）'}")


def main() -> int:
    print("=" * 74)
    print("  AGV 故障智能分析系统 —— 环境自检")
    print("=" * 74)

    check_python()
    check_cwd()
    check_packages()
    check_files()
    check_env()
    check_coppelia()
    check_ports()

    # ---- 汇总 ----
    failed = [(n, h) for n, ok, h in results if not ok]
    print("\n" + "=" * 74)
    if not failed:
        print(f"  {PASS} 全部通过，可以运行。")
        print("=" * 74)
        return 0

    print(f"  {FAIL} 发现 {len(failed)} 项问题：")
    print("=" * 74)
    for i, (name, hint) in enumerate(failed, 1):
        print(f"\n  {i}. {name}")
        if hint:
            for line in hint.split("\n"):
                print(f"     {line}")
    print("\n" + "=" * 74)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

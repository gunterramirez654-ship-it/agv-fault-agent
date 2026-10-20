#!/usr/bin/env bash
# ============================================================================
# run.sh —— AGV 项目统一启动入口
# ----------------------------------------------------------------------------
# 作用：自动使用项目 venv 里的 Python 和项目根目录，避免两类常见错误：
#   ① 用系统 python 运行 → 报 ModuleNotFoundError
#   ② 在其它目录运行     → 找不到 knowledge_base.json
#
# 用法：
#   ./run.sh doctor      环境自检（出问题先跑这个）
#   ./run.sh test        命令行全链路测试
#   ./run.sh web         启动 2D 可视化服务（http://127.0.0.1:8000）
#   ./run.sh scene       生成 CoppeliaSim 演示场景（3 台 AGV + 6 个障碍物）
#   ./run.sh bridge      启动 CoppeliaSim 3D 桥接（可追加参数）
#   ./run.sh sim         事件流仿真（若已安装）
#
# 示例：
#   ./run.sh bridge --duration 60 --cooldown 5
# ============================================================================
set -euo pipefail

# 切到脚本所在目录（项目根目录），无论从哪里调用
cd "$(dirname "$0")"

PY="./venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "[FATAL] 未找到虚拟环境：$PY"
    echo "        请确认项目根目录下存在 venv/，或重新创建："
    echo "        python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

CMD="${1:-}"
shift || true

case "$CMD" in
    doctor)
        exec "$PY" doctor.py "$@"
        ;;
    verify)
        exec "$PY" verify.py "$@"
        ;;
    test)
        exec "$PY" test_vector_store.py "$@"
        ;;
    web)
        exec "$PY" web_demo.py "$@"
        ;;
    scene)
        exec "$PY" prepare_scene.py "$@"
        ;;
    bridge)
        exec "$PY" coppelia_bridge.py "$@"
        ;;
    sim)
        if [ ! -f fleet_sim.py ]; then
            echo "[INFO] 尚未生成 fleet_sim.py（事件流仿真尚未实现）"
            exit 1
        fi
        exec "$PY" fleet_sim.py "$@"
        ;;
    *)
        echo "用法：./run.sh {verify|doctor|test|web|scene|bridge|sim} [参数...]"
        echo
        echo "  verify   全功能验证（交付/演示前跑这个）"
        echo "  doctor   环境自检（出问题先跑这个）"
        echo "  test     命令行全链路测试"
        echo "  web      启动 2D 可视化服务  → http://127.0.0.1:8000"
        echo "  scene    生成 CoppeliaSim 演示场景（3 台 AGV + 障碍物）"
        echo "  bridge   启动 CoppeliaSim 桥接（需先运行 scene 并打开场景）"
        echo "  sim      事件流仿真"
        echo
        echo "示例：./run.sh bridge --duration 60"
        exit 0
        ;;
esac

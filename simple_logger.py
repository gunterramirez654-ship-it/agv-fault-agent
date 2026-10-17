#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_logger.py —— AGV 具身智能项目 V1 POC 轻量化日志模块
================================================================================

定位：
    本模块是【独立的日志持久化工具】，只负责把业务事件写成 jsonl 文件。
    不与 agv_agent.py（LLM 组件）、vector_store.py（向量检索）产生任何耦合，
    也不被它们内部引用——按项目分层约定，日志调用全部发生在业务入口脚本
    （如 test_vector_store.py），由它把一次完整事件的全链路数据汇总后落盘。

设计取舍（V1 最小可用）：
    · 不使用 logging 重型框架，纯原生 Python 实现，零第三方依赖；
    · 存储格式为 jsonl（一行一条 JSON），便于后续用脚本逐行读取复盘；
    · 按【日期】分文件，同一天追加写入同一文件；
    · 提供全局单例 logger，外部直接导入使用，无需实例化。

明确不做（保持极简，避免 V1 过度设计）：
    ✗ 日志轮转 / 按大小切分    ✗ 压缩归档
    ✗ 告警通知                ✗ 控制台打印
    ✗ 多进程并发写入加锁      ✗ 日志级别过滤

使用方式：
    from simple_logger import logger

    logger.log_event({
        "event_text": "AGV在装载区视觉误识别货架，发生幻视",
        "similarity_score": 0.903,
        "risk_level": "高风险",
        "action": "AGV立即暂停，人工确认现场",
        "llm_analysis_report": "这里是大模型返回的分析报告",
        "cost_ms": 1256,
    })
    # log_event 会自动补上 timestamp 字段，无需调用方传入
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# 常量定义
# --------------------------------------------------------------------------- #
# 本模块所在目录：用于把 ./logs 固定定位到模块同级，
# 这样无论从哪个工作目录启动业务脚本，日志都落在同一个位置。
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 日志根目录（模块同级 ./logs，不存在时由本模块自动创建）
LOG_DIR = os.path.join(_MODULE_DIR, "logs")

# 日期格式：用于生成文件名，例如 20261013.jsonl
_DATE_FORMAT = "%Y%m%d"

# 时间戳格式：写入每条日志的 timestamp 字段，精确到微秒便于排查时序问题
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class SimpleLogger:
    """极简 jsonl 日志记录器。

    只提供一个对外方法 log_event()，负责：
        1) 自动补全 timestamp 字段；
        2) 按当天日期定位（必要时创建）日志文件；
        3) 以 UTF-8 + ensure_ascii=False 追加写入一行 JSON。

    使用建议：外部直接导入模块级单例 logger，不要自行实例化。
    """

    def __init__(self, log_dir: str = LOG_DIR) -> None:
        """初始化。

        参数：
            log_dir : 日志目录，默认模块同级 ./logs。

        说明：这里【不创建目录】，目录在首次写入时才按需创建，
        避免仅仅 import 本模块就在磁盘上留下空目录。
        """
        self.log_dir = log_dir
        self._current_date: Optional[str] = None   # 当前正在写入的日期（YYYYMMDD）
        self._current_path: Optional[str] = None   # 当前日志文件的完整路径

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _ensure_dir(self) -> None:
        """确保日志目录存在，不存在则创建（含多级父目录）。"""
        os.makedirs(self.log_dir, exist_ok=True)

    def _resolve_path(self) -> str:
        """返回今天应写入的日志文件路径。

        按日期切分文件：跨天运行时自动切换到新文件，
        路径解析结果缓存下来，避免每条日志都做一次日期格式化与拼接。
        """
        today = datetime.now().strftime(_DATE_FORMAT)
        if today != self._current_date:
            self._current_date = today
            self._current_path = os.path.join(self.log_dir, f"{today}.jsonl")
        # 类型收窄：上面分支已保证 _current_path 非空
        return str(self._current_path)

    # ------------------------------------------------------------------ #
    # 对外唯一方法
    # ------------------------------------------------------------------ #
    def log_event(self, event_data: Dict[str, Any]) -> bool:
        """把一条事件记录追加写入当天的 jsonl 日志文件。

        参数：
            event_data : 事件数据字典，字段由调用方自由组织（不做结构约束）。
                         本方法只做两件事：
                           · 复制一份数据并补上 timestamp 字段；
                           · 追加一行 JSON 到当天日志文件。
                         不会修改调用方传入的原始字典。

        返回：
            True  —— 写入成功
            False —— 写入失败（异常信息通过 stderr 输出，不向调用方抛异常）

        设计说明：
            日志属于旁路能力，不应因为磁盘满、权限不足等问题中断 AGV 主业务链路，
            因此这里捕获 IO 类异常并返回 False，由调用方决定是否忽略。
            但【序列化失败】属于代码缺陷（传入了不可 JSON 化的对象），
            这种情况会抛出 TypeError，便于在联调阶段尽早暴露问题。
        """
        # ---- 1) 参数校验：必须是字典 ----
        if not isinstance(event_data, dict):
            raise TypeError(
                f"event_data 必须是 dict，当前为：{type(event_data).__name__}"
            )

        # ---- 2) 复制并补全时间戳（不修改调用方原始数据）----
        record: Dict[str, Any] = dict(event_data)
        # 调用方已自带 timestamp 时予以保留，避免覆盖业务侧更精确的时间
        record.setdefault("timestamp", datetime.now().strftime(_TIMESTAMP_FORMAT))

        # ---- 3) 序列化：ensure_ascii=False 保证中文可读 ----
        # 说明：这里【不做 default=str 兜底】，因为静默把对象转成字符串会掩盖
        #      数据建模问题，不利于后续复盘；宁可抛 TypeError 让问题早暴露。
        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"日志内容无法序列化为 JSON：{type(exc).__name__}: {exc}"
            ) from exc

        # ---- 4) 追加写入：目录按需创建，文件以 UTF-8 追加 ----
        try:
            self._ensure_dir()
            path = self._resolve_path()
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except OSError as exc:
            # 磁盘满、无写权限、路径被占用等：不中断业务，仅提示
            print(
                f"[simple_logger] 日志写入失败：{type(exc).__name__}: {exc}",
                file=__import__("sys").stderr,
            )
            return False

        return True


# --------------------------------------------------------------------------- #
# 模块级单例：外部直接 from simple_logger import logger 即可使用
# --------------------------------------------------------------------------- #
logger = SimpleLogger()
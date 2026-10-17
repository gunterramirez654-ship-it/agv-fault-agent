#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vector_store.py —— AGV 巡检小车：故障知识库向量检索组件（V1 POC）
================================================================================

核心能力（本项目创新点）：
    区域优先召回（Area-First Retrieval）
        · 传入 area_id 时，优先在本区域案例内做向量检索；
        · 本区域命中不足 top_k 时，再从其它区域按相似度补齐；
        · 这是向量库层面的召回策略，不是 LLM 层的提示词技巧。

数据来源：
    项目根目录 knowledge_base.json（16 条 AGV 故障案例，4 个区域各 4 条），
    代码内【不硬编码任何案例】，全部从外部 JSON 读取。

技术选型：
    · 向量化：chromadb 自带 ONNXMiniLM_L6_V2（all-MiniLM-L6-v2，384 维，离线）
    · 相似度：手写余弦相似度（内存计算，不依赖 Chroma 持久化向量库）
    · 检索  ：内存暴力 KNN（POC 数据量小，几百条以内延迟可忽略）

============================= 历史故障与防复发措施 =============================
本文件曾因手工注释掉 import 报 "NameError: name 'load_dotenv' is not defined"。
因此做了三件事防止复发：
    1) 完全不用 python-dotenv，改用标准库自己解析 .env（_load_dotenv_lightweight）
    2) 所有 import 集中在文件头部，且严禁注释掉（注释 import 但保留调用必炸）
    3) 依赖缺失时给出明确的 pip 提示，而不是抛裸 NameError
==============================================================================

依赖：
    pip install chromadb onnxruntime

模型目录（离线部署，6 个文件缺一不可）：
    onnx_models/all-MiniLM-L6-v2/onnx/
        ├── model.onnx                 ← ONNX 权重
        ├── tokenizer.json             ← 分词器
        ├── config.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── vocab.txt
    本模块的向量化由 bge_embedding.py 提供（BGE 中文模型，512 维），
    会自动探测模型目录，也可用环境变量 BGE_MODEL_DIR 显式指定。

使用方式：
    from vector_store import VectorStore

    vs = VectorStore()                        # 自动加载 knowledge_base.json
    results = vs.search("AGV视觉误识别货架", top_k=3, area_id="region_RAN")
    # 每条结果包含：score / similarity / fault_id / fault_desc / text
    #              / area_id / risk_level / action
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 常量与路径
# --------------------------------------------------------------------------- #
# 本模块所在目录（即项目根目录）。所有相对路径都以它为基准，
# 这样无论从哪个工作目录启动脚本，都能找到知识库与模型。
_MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# 知识库默认路径（可用构造参数 kb_path 或环境变量 KB_PATH 覆盖）
DEFAULT_KB_PATH: str = os.path.join(_MODULE_DIR, "knowledge_base.json")

# 向量模型名与维度
# 已从 all-MiniLM-L6-v2（384维，英文词表，中文全变 [UNK]）换为
# BGE 中文模型 bge-small-zh-v1.5（512维，中文分词正常）
MODEL_NAME: str = "bge-small-zh-v1.5"
EMBEDDING_DIM: int = 512

# 知识库每条记录的必填字段
REQUIRED_FIELDS: Tuple[str, ...] = (
    "fault_id",
    "fault_desc",
    "area_id",
    "risk_level",
    "action",
)

# 支持的作业区域（用于 area_id 合法性校验）
SUPPORTED_AREAS: Tuple[str, ...] = (
    "region_RAN",  # 装载区
    "region_RAZ",  # 缓存区
    "region_RDA",  # 充电区
    "region_RAB",  # 避让区
)

# ONNX 模型必需的 6 个文件（与 chromadb 内部校验清单一致）
_REQUIRED_MODEL_FILES: Tuple[str, ...] = (
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


# --------------------------------------------------------------------------- #
# 自定义异常
# --------------------------------------------------------------------------- #
class VectorStoreError(RuntimeError):
    """本模块所有异常的基类，方便调用方一次性捕获。"""


class KnowledgeBaseError(VectorStoreError):
    """知识库异常：文件不存在、JSON 非法、记录缺字段、检索参数非法。"""


class EmbeddingError(VectorStoreError):
    """向量化异常：依赖缺失、本地模型文件缺失或加载失败。"""


# --------------------------------------------------------------------------- #
# .env 轻量解析（纯标准库，不引入 python-dotenv 依赖）
# --------------------------------------------------------------------------- #
def _load_dotenv_lightweight() -> None:
    """读取项目根目录 .env，把 KEY=VALUE 注入 os.environ。

    为什么不用 python-dotenv：
        · 少一个第三方依赖，部署更简单；
        · 避免"装了但 import 被注释掉"导致的 NameError 复发。

    行为约定：
        · 只支持 KEY=VALUE，忽略空行与 # 注释行；
        · 已存在的真实环境变量优先，不被 .env 覆盖；
        · 文件不存在或读取失败时静默跳过，不影响主流程。
    """
    env_path = os.path.join(_MODULE_DIR, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                # 去掉行内注释与首尾引号
                value = value.split("#", 1)[0].strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class FaultRecord:
    """知识库中的一条故障案例（向量与元数据绑定存放）。

    字段：
        fault_id   : 故障唯一编号
        fault_desc : 故障文本描述（向量化输入）
        area_id    : 所属作业区域
        risk_level : 该案例的参考风险等级（原样保留，不做任何改写）
        action     : 该案例的参考处置动作（原样保留，不做任何改写）
        embedding  : fault_desc 对应的归一化向量
    """

    fault_id: str
    fault_desc: str
    area_id: str
    risk_level: str
    action: str
    embedding: List[float]

    def to_result(self, score: float) -> Dict[str, Any]:
        """转换为对外检索结果字典。

        同时提供两套键名，避免上下游字段对不上：
            score / similarity : 同一数值（余弦相似度，越大越相似）
            fault_desc / text  : 同一文本（故障描述）
        这样 agv_agent.analyze() 需要的 similarity / text 也能直接取到。
        """
        return {
            # 主键名（业务与日志使用）
            "score": float(score),
            "fault_id": self.fault_id,
            "fault_desc": self.fault_desc,
            "area_id": self.area_id,
            "risk_level": self.risk_level,
            "action": self.action,
            # 兼容别名（agv_agent 使用）
            "similarity": float(score),
            "text": self.fault_desc,
        }


# --------------------------------------------------------------------------- #
# 余弦相似度（手写，不依赖 sklearn / scipy）
# --------------------------------------------------------------------------- #
def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度，返回 [-1, 1] 的浮点数。

    公式：cos = (a · b) / (|a| * |b|)

    边界处理：
        · 长度不一致时按较短长度对齐，避免数据版本不一致直接报错；
        · 任一向量模长为 0 时返回 0.0，避免除零崩溃。
    """
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(length):
        a = float(vec_a[i])
        b = float(vec_b[i])
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# --------------------------------------------------------------------------- #
# 向量化：BGE 中文模型（离线 ONNX）
# --------------------------------------------------------------------------- #
# 实现细节集中在 bge_embedding.py，本模块只负责调用。换模型的背景与实测
# 对比数据见该文件头部说明（原 MiniLM 中文全为 [UNK]，正确项与错误项分差
# 仅 0.0002；换 BGE 后分差 0.21）。
def build_embedding_fn() -> Callable[..., List[float]]:
    """构造向量化函数（BGE 中文模型，512 维，已归一化）。

    返回的函数支持 is_query 关键字参数：
        fn(text)                 -> 文档（不加 BGE 指令前缀）
        fn(text, is_query=True)  -> 查询（自动加指令前缀）

    异常：
        EmbeddingError : BGE 模型文件缺失或加载失败
    """
    try:
        from bge_embedding import build_embedding_fn as _build_bge
    except ImportError as exc:
        raise EmbeddingError(
            "向量化不可用：缺少 bge_embedding 模块或其依赖。\n"
            "    请确认 bge_embedding.py 与本模块同目录，并执行：\n"
            "    pip install onnxruntime tokenizers numpy"
        ) from exc

    try:
        return _build_bge()
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"BGE 模型加载失败：{type(exc).__name__}: {exc}") from exc

# --------------------------------------------------------------------------- #
# 向量检索组件
# --------------------------------------------------------------------------- #
class VectorStore:
    """AGV 故障知识库向量检索组件（内存版，区域优先召回）。

    使用示例：
        vs = VectorStore()
        results = vs.search("AGV在装载区视觉误识别货架", top_k=3, area_id="region_RAN")
        for item in results:
            print(item["score"], item["area_id"], item["fault_desc"])

    说明：
        · 初始化即自动加载知识库并完成向量化，不提供 add 接口；
        · 数据全部来自外部 JSON，代码中不硬编码任何案例。
    """

    def __init__(
        self,
        kb_path: Optional[str] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """加载知识库并建立内存向量索引。

        参数：
            kb_path      : 知识库 JSON 路径；不传时取环境变量 KB_PATH，
                           再退化为项目根目录 knowledge_base.json
            embedding_fn : 自定义向量化函数 fn(text) -> List[float]（可选，便于测试）；
                           不传时使用本地 ONNX 模型

        异常：
            KnowledgeBaseError : 文件不存在、JSON 非法、记录缺字段
            EmbeddingError     : 向量化模型不可用
        """
        # 解析知识库路径：构造参数 > 环境变量 > 默认路径
        if kb_path:
            self.kb_path: str = kb_path
        else:
            _load_dotenv_lightweight()
            self.kb_path = (os.getenv("KB_PATH") or "").strip() or DEFAULT_KB_PATH

        self.records: List[FaultRecord] = []
        self._embedding_fn: Optional[Callable[[str], List[float]]] = embedding_fn
        # 按区域建分区索引，避免每次检索都全表过滤
        self._area_index: Dict[str, List[FaultRecord]] = {}

        # 初始化即完成「加载知识库 → 向量化 → 建索引」，失败直接抛错不静默
        self._load_knowledge_base()

    # ====================================================================== #
    # 逻辑段一：加载知识库
    # ====================================================================== #
    def _load_knowledge_base(self) -> None:
        """读取并校验 knowledge_base.json，解析为 FaultRecord 列表。"""
        # ---- 1) 文件存在性 ----
        if not os.path.isfile(self.kb_path):
            raise KnowledgeBaseError(
                f"故障知识库文件不存在：{self.kb_path}\n"
                "    请确认 knowledge_base.json 已放在项目根目录，"
                "或用参数 kb_path / 环境变量 KB_PATH 指定正确路径。"
            )

        # ---- 2) 读文本（UTF-8，保证中文正常）----
        try:
            with open(self.kb_path, "r", encoding="utf-8") as fp:
                raw_text = fp.read()
        except OSError as exc:
            raise KnowledgeBaseError(
                f"故障知识库读取失败：{self.kb_path}\n"
                f"    原因：{type(exc).__name__}: {exc}"
            ) from exc

        # ---- 3) JSON 解析（格式错误给出行列位置）----
        try:
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"故障知识库 JSON 格式非法：{self.kb_path}\n"
                f"    位置：第 {exc.lineno} 行，第 {exc.colno} 列\n"
                f"    原因：{exc.msg}\n"
                "    常见原因：多余逗号、缺少引号、括号不匹配。"
            ) from exc

        # ---- 4) 顶层结构校验 ----
        if not isinstance(raw_data, list):
            raise KnowledgeBaseError(
                f"故障知识库结构非法：{self.kb_path}\n"
                f"    期望顶层为 JSON 数组，实际为 {type(raw_data).__name__}。"
            )
        if not raw_data:
            raise KnowledgeBaseError(f"故障知识库为空数组：{self.kb_path}")

        # ---- 5) 逐条校验（先全部校验再向量化，避免半途失败留下脏状态）----
        normalized = [
            self._validate_record(item, index)
            for index, item in enumerate(raw_data, start=1)
        ]

        # ---- 6) 向量化并建索引 ----
        self._build_index(normalized)

    @staticmethod
    def _validate_record(item: Any, index: int) -> Dict[str, str]:
        """校验单条记录并归一化为字符串字典。

        校验规则：
            · 必须是 JSON 对象；
            · 必须包含 REQUIRED_FIELDS 中全部字段；
            · fault_desc 不能为空白（它是向量化输入）。

        注意：risk_level / action 只做存在性与字符串化处理，
        原样保留业务语义，不做任何改写或推断。
        """
        if not isinstance(item, dict):
            raise KnowledgeBaseError(
                f"知识库第 {index} 条记录非法：期望 JSON 对象，"
                f"实际为 {type(item).__name__}。"
            )

        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise KnowledgeBaseError(
                f"知识库第 {index} 条记录缺少必需字段：{', '.join(missing)}\n"
                f"    必需字段：{', '.join(REQUIRED_FIELDS)}\n"
                f"    实际字段：{', '.join(str(k) for k in item.keys()) or '（无）'}"
            )

        record = {field: str(item.get(field) or "").strip() for field in REQUIRED_FIELDS}
        if not record["fault_desc"]:
            raise KnowledgeBaseError(
                f"知识库第 {index} 条记录"
                f"（fault_id={record['fault_id'] or '未知'}）的 fault_desc 为空，"
                "无法生成向量。"
            )
        return record

    # ====================================================================== #
    # 逻辑段二：向量插入（逐条向量化，并建立区域分区索引）
    # ====================================================================== #
    def _build_index(self, records: Sequence[Dict[str, str]]) -> None:
        """对每条 fault_desc 生成向量，存入内存并建立按区域的分区索引。"""
        self.records = []
        self._area_index = {area: [] for area in SUPPORTED_AREAS}

        for record in records:
            item = FaultRecord(
                fault_id=record["fault_id"],
                fault_desc=record["fault_desc"],
                area_id=record["area_id"],
                risk_level=record["risk_level"],
                action=record["action"],
                embedding=self.embedding(record["fault_desc"]),
            )
            self.records.append(item)
            # 分区索引：area_id 不在预设列表里也不丢数据，仍可被全局召回
            self._area_index.setdefault(item.area_id, []).append(item)

    def embedding(self, text: str, is_query: bool = False) -> List[float]:
        """把一段文本转成向量（BGE 中文模型，512 维，已归一化）。

        参数：
            text     : 待向量化文本（知识库加载与查询检索共用此入口）
            is_query : True = 查询（加 BGE 指令前缀）；False = 文档（默认）

        说明：
            BGE 要求查询加指令前缀、文档不加。这个区分由本参数控制，
            加错会显著降低检索效果，因此默认值取"文档"，
            查询侧必须显式传 is_query=True（见 search 方法）。

        异常：
            KnowledgeBaseError : text 为空白
            EmbeddingError     : 向量化模型不可用
        """
        if not isinstance(text, str) or not text.strip():
            raise KnowledgeBaseError("待向量化文本必须是非空字符串。")

        if self._embedding_fn is None:
            self._embedding_fn = build_embedding_fn()
        return [float(v) for v in self._embedding_fn(text, is_query=is_query)]

    # ====================================================================== #
    # 逻辑段三：检索（区域优先 → 全局兜底 → 组装结果）
    # ====================================================================== #
    def search(
        self,
        query_text: str,
        top_k: int = 3,
        area_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """区域优先 + 全局兜底的余弦相似度检索。

        召回流程：
            area_id 不传（None）：
                对全部案例做全局检索，按相似度降序取 top_k。

            area_id 传入：
                ① 先在 area_id 匹配的案例内检索，按相似度降序取前 top_k；
                ② 若本区域命中不足 top_k，再从其它区域按相似度从高到低补齐；
                ③ 合并后整体按相似度降序重排（否则会出现"后面比前面更像"）。

        参数：
            query_text : 现场事件文本
            top_k      : 期望返回条数，默认 3
            area_id    : 区域 ID（region_RAN / region_RAZ / region_RDA / region_RAB）；
                         为 None 时执行全局检索

        返回：
            List[Dict]，按 score 降序，每条包含：
                score / similarity : 余弦相似度（越大越相似）
                fault_id           : 故障编号
                fault_desc / text  : 故障描述
                area_id            : 所属区域
                risk_level         : 该案例的参考风险等级（原样返回）
                action             : 该案例的参考处置动作（原样返回）

        异常：
            KnowledgeBaseError : query_text / top_k / area_id 参数非法

        说明：
            本方法只做相似度计算与区域优先排序，不做权重打分、不做 rerank、
            不做阈值过滤；也不对 risk_level / action 做任何判断与改写。
        """
        # ---- 参数校验 ----
        if not isinstance(query_text, str) or not query_text.strip():
            raise KnowledgeBaseError("query_text 必须是非空字符串。")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise KnowledgeBaseError(f"top_k 必须是正整数，当前为：{top_k!r}")
        if area_id is not None:
            if not isinstance(area_id, str) or not area_id.strip():
                raise KnowledgeBaseError(
                    f"area_id 必须是非空字符串或 None，当前为：{area_id!r}"
                )
            area_id = area_id.strip()
            if area_id not in SUPPORTED_AREAS:
                raise KnowledgeBaseError(
                    f"不支持的区域 ID：{area_id!r}\n"
                    f"    支持的区域：{', '.join(SUPPORTED_AREAS)}"
                )

        if not self.records:
            return []

        # ---- 1) 查询文本向量化 ----
        # 查询必须传 is_query=True：BGE 需要给查询加指令前缀（文档不加）
        query_vector = self.embedding(query_text, is_query=True)

        # ---- 2) 全局检索（area_id 不传时走这里；也是区域不足时的兜底来源）----
        if area_id is None:
            scored = self._score(query_vector, self.records)
            return [rec.to_result(score) for score, rec in scored[:top_k]]

        # ---- 3) 区域优先：先在本区域案例内检索 ----
        same_area = self._area_index.get(area_id, [])
        picked = self._score(query_vector, same_area)[:top_k]
        picked_ids = {rec.fault_id for _, rec in picked}

        # ---- 4) 全局兜底：不足 top_k 时从其它区域补齐差额 ----
        if len(picked) < top_k:
            others = [r for r in self.records if r.fault_id not in picked_ids]
            for score, rec in self._score(query_vector, others):
                if len(picked) >= top_k:
                    break
                picked.append((score, rec))
            # 合并后必须整体重排，保证严格降序
            picked.sort(key=lambda pair: pair[0], reverse=True)

        # ---- 5) 组装结果：相似度 + 完整元数据 ----
        return [rec.to_result(score) for score, rec in picked]

    @staticmethod
    def _score(
        query_vector: Sequence[float],
        records: Sequence[FaultRecord],
    ) -> List[Tuple[float, FaultRecord]]:
        """对给定案例集合计算余弦相似度，返回按相似度降序的 (score, record) 列表。"""
        scored: List[Tuple[float, FaultRecord]] = []
        for record in records:
            if not record.embedding:
                continue
            scored.append((cosine_similarity(query_vector, record.embedding), record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    # ------------------------------------------------------------------ #
    # 自检辅助方法（供业务入口打印，不参与召回逻辑）
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """返回已加载的故障案例总数。"""
        return len(self.records)

    def area_distribution(self) -> Dict[str, int]:
        """返回各区域的案例数量分布，便于启动时自检知识库覆盖度。"""
        distribution: Dict[str, int] = {
            area: len(self._area_index.get(area, [])) for area in SUPPORTED_AREAS
        }
        # 把不在预设列表里的区域也带上，避免统计口径丢数据
        for area, items in self._area_index.items():
            if area not in distribution:
                distribution[area] = len(items)
        return distribution
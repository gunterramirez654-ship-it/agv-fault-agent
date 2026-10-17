#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bge_embedding.py —— BGE 中文向量化模块（离线 ONNX）
================================================================================

用途：
    为 vector_store.py 提供中文向量化能力。模型使用 BAAI/bge-small-zh-v1.5 的
    ONNX 版本，纯 CPU 推理，不联网、不需要 PyTorch。

--------------------------------------------------------------------------
为什么换掉原来的 all-MiniLM-L6-v2（实测结论，非推测）
--------------------------------------------------------------------------
    原模型是【英文专用词表】：vocab 仅 30522，其中含中文字符的 token 只有
    488 个且全部是 "##接续子词" 形式，无法独立成词。实测分词结果：

        '视觉幻视'  → ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]']
        'relay protection' → ['relay', 'protection']        ← 英文正常

    也就是说模型拿到的是一串"未知符号"，根本没在理解中文语义，检索排序
    接近随机。实测正确项与最佳错误项的相似度分差仅 0.0002。

    换成 BGE-small-zh-v1.5（vocab 21128，含中文 token 14642 个）后：

        '视觉幻视'  → ['[CLS]', '视', '觉', '幻', '视', '[SEP]']   UNK=0

    实测正确项与最佳错误项的分差提升到 0.21（约 1000 倍）。

--------------------------------------------------------------------------
================= BGE 的两个关键使用约定（必须遵守）=================
--------------------------------------------------------------------------
    ① 池化方式：取 [CLS] 位置向量（last_hidden_state[:, 0]），
       不是 mean pooling。加错会显著降低效果。

    ② 查询前缀：查询文本需要加指令前缀，【文档不加】：
           前缀 = "为这个句子生成表示以用于检索相关文章："
       如果给文档也加了前缀（或查询不加），效果可能比原模型还差。

--------------------------------------------------------------------------
模型文件要求（离线部署，人工拷贝到设备）
--------------------------------------------------------------------------
    <模型根目录>/bge-small-zh-v1.5/onnx/
        ├── model.onnx              ← 约 90MB，单文件自带权重
        ├── tokenizer.json          ← 约 0.42MB（注意：不是 0.68MB 的 MiniLM 词表）
        ├── config.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── vocab.txt

    本模块会自动探测多个候选目录，也可用环境变量 BGE_MODEL_DIR 显式指定。

--------------------------------------------------------------------------
依赖
--------------------------------------------------------------------------
    pip install onnxruntime tokenizers numpy
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
MODEL_NAME = "bge-small-zh-v1.5"

# BGE 官方建议的查询指令前缀（只加在查询上，文档不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模型运行必需的文件（与目录内实际文件对应）
_REQUIRED_FILES = (
    "model.onnx",
    "tokenizer.json",
    "config.json",
)

# 最大序列长度（BGE 支持 512；POC 场景文本都很短，超出即截断）
_MAX_LENGTH = 512


class BGEEmbeddingError(RuntimeError):
    """BGE 向量化异常：依赖缺失、模型文件缺失或加载失败。"""


# --------------------------------------------------------------------------- #
# 模型目录定位
# --------------------------------------------------------------------------- #
def _find_model_dir(module_dir: str, override: Optional[str] = None) -> str:
    """定位 BGE 的 onnx 目录，返回绝对路径。

    查找顺序（逐个尝试，找到完整目录即返回）：
        1) 参数 override 指定
        2) 环境变量 BGE_MODEL_DIR
        3) <项目根>/onnx_models/bge-small-zh-v1.5/onnx
        4) <项目根>/models/bge-small-zh-v1.5/onnx
        5) <项目根>/bge-small-zh-v1.5/onnx

    异常：
        BGEEmbeddingError : 所有候选目录都不完整（会列出全部尝试过的路径）
    """
    roots: List[str] = []

    if override:
        roots.append(os.path.abspath(os.path.expanduser(override)))

    env_dir = (os.getenv("BGE_MODEL_DIR") or "").strip()
    if env_dir:
        env_dir = os.path.expanduser(env_dir)
        if not os.path.isabs(env_dir):
            env_dir = os.path.join(module_dir, env_dir)
        roots.append(os.path.abspath(env_dir))

    roots.extend([
        os.path.join(module_dir, "onnx_models", MODEL_NAME, "onnx"),
        os.path.join(module_dir, "models", MODEL_NAME, "onnx"),
        os.path.join(module_dir, MODEL_NAME, "onnx"),
    ])

    tried: List[str] = []
    for candidate in roots:
        tried.append(candidate)
        if all(os.path.isfile(os.path.join(candidate, f)) for f in _REQUIRED_FILES):
            return candidate

    raise BGEEmbeddingError(
        "【离线部署】未找到完整的 BGE ONNX 模型。\n"
        f"    需要以下文件放在同一个目录：{', '.join(_REQUIRED_FILES)}\n"
        "    已尝试过的目录：\n"
        + "".join(f"        - {p}\n" for p in tried)
        + "    处理办法：把模型文件放到上述任一目录，或设置环境变量：\n"
        f"        BGE_MODEL_DIR=/绝对路径/{MODEL_NAME}/onnx"
    )


# --------------------------------------------------------------------------- #
# BGE 向量化器
# --------------------------------------------------------------------------- #
class BGEEmbedding:
    """BGE 中文向量化器（离线 ONNX，512 维，已归一化）。

    使用方式：
        emb = BGEEmbedding()
        doc_vecs  = emb.encode_documents(["故障描述1", "故障描述2"])   # 不加前缀
        query_vec = emb.encode_query("现场事件文本")                   # 自动加前缀

    说明：
        · 初始化时做一次预热推理，模型损坏会在启动阶段就报错；
        · 输出统一为 List[List[float]] / List[float]，避免 numpy 类型向下游扩散。
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        """加载模型与分词器。

        参数：
            model_dir : 可选，直接指定 onnx 目录；不传时自动探测

        异常：
            BGEEmbeddingError : 依赖缺失、模型缺失或加载失败
        """
        # ---- 依赖检测：缺失时给明确 pip 提示，而不是裸 ImportError ----
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise BGEEmbeddingError(
                "BGE 向量化依赖缺失，请执行：\n"
                "    pip install onnxruntime tokenizers numpy"
            ) from exc

        self._np = np

        # ---- 定位模型目录 ----
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir: str = _find_model_dir(module_dir, override=model_dir)

        # ---- 加载 ONNX 会话与分词器 ----
        try:
            self.session = ort.InferenceSession(
                os.path.join(self.model_dir, "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            self.tokenizer = Tokenizer.from_file(
                os.path.join(self.model_dir, "tokenizer.json")
            )
            self.tokenizer.enable_truncation(max_length=_MAX_LENGTH)
        except Exception as exc:  # noqa: BLE001
            raise BGEEmbeddingError(
                f"BGE 模型加载失败：{type(exc).__name__}: {exc}\n"
                f"    模型目录：{self.model_dir}"
            ) from exc

        # 记录输入/输出名，兼容不同导出版本
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]

        # ---- 预热：确认可推理，并取得向量维度 ----
        probe = self._forward(["warmup"])
        self.dimension: int = len(probe[0])

    # ------------------------------------------------------------------ #
    # 内部：ONNX 前向推理
    # ------------------------------------------------------------------ #
    def _forward(self, texts: Sequence[str]) -> List[List[float]]:
        """把一批文本转成归一化向量（内部方法，不处理前缀）。"""
        np = self._np

        encodings = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.array(
            [e.ids + [0] * (max_len - len(e.ids)) for e in encodings],
            dtype=np.int64,
        )
        attention_mask = np.array(
            [e.attention_mask + [0] * (max_len - len(e.attention_mask))
             for e in encodings],
            dtype=np.int64,
        )

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 部分导出会要求 token_type_ids，按需补全为 0
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feeds)

        # 池化：优先用模型自带的 sentence_embedding，否则取 [CLS] 位置
        if "sentence_embedding" in self._output_names:
            vectors = outputs[self._output_names.index("sentence_embedding")]
        else:
            hidden = outputs[self._output_names.index("last_hidden_state")]
            vectors = hidden[:, 0]      # CLS pooling（BGE 标准做法）

        # L2 归一化：余弦相似度计算的前提
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)
        return [[float(x) for x in row] for row in vectors]

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """向量化【文档】（知识库条目）——不加指令前缀。"""
        if not texts:
            return []
        return self._forward(list(texts))

    def encode_query(self, text: str) -> List[float]:
        """向量化【查询】（现场事件）——自动加 BGE 指令前缀。"""
        return self._forward([QUERY_INSTRUCTION + text])[0]


# --------------------------------------------------------------------------- #
# 供 vector_store.py 使用的构造函数
# --------------------------------------------------------------------------- #
def build_embedding_fn() -> Callable[..., List[float]]:
    """构造向量化函数，签名兼容 vector_store.py 的调用方式。

    返回的函数带一个可选关键字参数 is_query：
        fn("文本")                 -> 按【文档】处理（不加前缀）
        fn("文本", is_query=True)  -> 按【查询】处理（加前缀）

    这样 vector_store.py 只需极小改动即可接入 BGE 中文模型。
    """
    embedder = BGEEmbedding()

    def _embed(text: str, is_query: bool = False) -> List[float]:
        if is_query:
            return embedder.encode_query(text)
        return embedder.encode_documents([text])[0]

    # 附带维度信息，便于上层核对（BGE-small-zh 为 512）
    _embed.dimension = embedder.dimension       # type: ignore[attr-defined]
    return _embed

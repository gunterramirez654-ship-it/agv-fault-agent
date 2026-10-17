'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-14 02:34:36
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-14 02:39:53
FilePath: /agv_embodied_project/onnx_embedding.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# onnx_embedding.py
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class ONNXEmbedding:
    def __init__(self, onnx_path: str, tokenizer_path: str):
        # 加载onnx推理session，M1默认cpu即可，后面可以换silicon加速
        self.session = ort.InferenceSession(onnx_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str):
        """输入单段文本，输出归一化向量"""
        # 分词编码
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        # ========== 新增这一行：补上 token_type_ids，全部填0 ==========
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # ONNX前向推理，把三个参数全部传进去
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )
        last_hidden_state = outputs[0]  # shape: [1, seq_len, hidden_size]

        # mean pooling 均值池化，all-MiniLM标准操作
        mask_expanded = attention_mask[0].reshape(-1, 1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state[0] * mask_expanded, axis=0)
        sum_mask = np.sum(mask_expanded)
        embedding = sum_embeddings / sum_mask

        # L2归一化，向量检索必须归一化，余弦相似度计算更稳定
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding


    def encode_batch(self, texts: list[str]):
        # 批量接口，给Chroma用
        return [self.encode(t) for t in texts]


# 本地测试入口
if __name__ == "__main__":
    model = ONNXEmbedding(
        onnx_path="onnx_models/all-MiniLM-L6-v2/onnx/model.onnx",
        tokenizer_path="onnx_models/all-MiniLM-L6-v2/onnx/tokenizer.json"
    )
    vec1 = model.encode("AGV在装载货架，发生侧翻")
    vec2 = model.encode("猫咪在晒太阳")
    sim = np.dot(vec1, vec2)
    print(f"向量维度：{vec1.shape}")
    print(f"相似度：{sim:.4f}")

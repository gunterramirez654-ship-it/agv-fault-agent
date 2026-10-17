import chromadb
from onnx_embedding import ONNXEmbedding
import numpy as np

class AgvVectorStore:
    def __init__(self, embedding: ONNXEmbedding, persist_path: str = "./agv_vector_db"):
        self.embedding = embedding
        self.client = chromadb.PersistentClient(path=persist_path)
        # 创建集合，强制指定余弦距离，适配all-MiniLM
        try:
            self.client.delete_collection(name="agv_policy")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="agv_policy",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        vec = self.embedding.encode(text)
        embedding_list = vec.tolist()
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding_list],
            documents=[text],
            metadatas=[metadata] if metadata else None
        )

    def search(self, query_text: str, top_k: int = 1):
        query_vec = self.embedding.encode(query_text)
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k
        )
        return res


if __name__ == "__main__":
    # ========= 裸向量距离调试，绕开Chroma =========
    embed = ONNXEmbedding(
        onnx_path="onnx_models/all-MiniLM-L6-v2/onnx/model.onnx",
        tokenizer_path="onnx_models/all-MiniLM-L6-v2/onnx/tokenizer.json"
    )

    text1 = "AGV装载货架发生偏移，高风险，立即暂停，人工确认现场"
    text2 = "AGV前方障碍物，中风险，减速行驶，扫描周边环境，上传异常日志"
    text3 = "AGV电量偏低，低风险，继续作业，等待空闲时段回充"
    query_text = "AGV装载货架发生偏移"

    v1 = embed.encode(text1)
    v2 = embed.encode(text2)
    v3 = embed.encode(text3)
    vq = embed.encode(query_text)

    def calc_dist(a, b):
        return np.linalg.norm(a - b)

    print("===== 向量距离调试 =====")
    print(f"query vs text1(货架偏移): {calc_dist(vq, v1):.4f}")
    print(f"query vs text2(障碍物): {calc_dist(vq, v2):.4f}")
    print(f"query vs text3(电量偏低): {calc_dist(vq, v3):.4f}")
    # =====================================================

    vs = AgvVectorStore(embed)

    # 写入几条AGV故障策略
    vs.add_document("doc1", "AGV装载货架发生偏移，高风险，立即暂停，人工确认现场", {"level": "high"})
    vs.add_document("doc2", "AGV前方障碍物，中风险，减速行驶，扫描周边环境，上传异常日志", {"level": "mid"})
    vs.add_document("doc3", "AGV电量偏低，低风险，继续作业，等待空闲时段回充", {"level": "low"})

    # ========= 在这里随便改你想要测试的查询句子 =========
    query = "AGV装载货架发生偏移"
    result = vs.search(query, top_k=3)

    print(f"\n✅ 查询文本：{query}")
    print("===== 检索结果 top3 =====")
    for i in range(len(result["documents"][0])):
        print(f"👉 第{i+1}条 距离：{result['distances'][0][i]:.4f} | 风险等级：{result['metadatas'][0][i]['level']}")
        print(f"    策略内容：{result['documents'][0][i]}")

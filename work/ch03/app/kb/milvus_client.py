"""Milvus Standalone(HTTP 19530)客户端:pymilvus 的 MilvusClient 高层接口。

主键 id = MySQL knowledge_chunks.id(auto_id=False,由我方提供),使重跑按 id upsert 天然幂等。
COSINE 相似度:Standalone 下 search 回的 distance 即相似度(越大越像),阈值直接比 score。
"""
from pymilvus import DataType, MilvusClient

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024
MAX_QUESTION = 2048
MAX_ANSWER = 8192

_CLIENT: MilvusClient | None = None


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Standalone);传 uri 返回独立客户端(测试/临时)。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等:已存在则确保加载;不存在则按 schema 建集合 + AUTOINDEX/COSINE 索引 + 加载。"""
    if client.has_collection(collection):
        client.load_collection(collection)
        return
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("question", DataType.VARCHAR, max_length=MAX_QUESTION)
    schema.add_field("answer", DataType.VARCHAR, max_length=MAX_ANSWER)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection, schema=schema, index_params=index_params)
    client.load_collection(collection)


def _fit(text: str, limit: int) -> str:
    """按 UTF-8 字节数裁剪:Milvus VARCHAR 的 max_length 计的是字节(中文一字 3 字节),
    只按字符数裁会在多字节文本上被 Milvus 以 code=1100 整批拒绝。截断处丢掉半个字符即可。"""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    """按主键 upsert(同 id 覆盖,天然幂等)。超 max_length 的字段先裁剪,免得整批被 Milvus 拒。"""
    if not rows:
        return
    fitted = [
        {**r, "question": _fit(r["question"], MAX_QUESTION), "answer": _fit(r["answer"], MAX_ANSWER)}
        for r in rows
    ]
    client.upsert(collection, fitted)


def search(client: MilvusClient, vector: list[float], top_k: int,
           collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[vector], anns_field="vector", limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {"id": h["id"], "score": float(h["distance"]),
         "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
        for h in res[0]
    ]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:Standalone 新 upsert 的数据须 flush 后才稳定可检索。"""
    client.flush(collection)


def drop(client: MilvusClient, collection: str = COLLECTION) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)

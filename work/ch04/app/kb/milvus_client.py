"""Milvus Standalone(HTTP 19530)客户端:pymilvus MilvusClient 高层接口。

ch04 schema = dense(COSINE)+ text(chinese analyzer,BM25 Function 输入)+ sparse(BM25 输出)
+ question/answer/section_path/content_type/category 标量。
主键 id = MySQL knowledge_chunks.id(auto_id=False),重跑按 id upsert,天然幂等。

COSINE 相似度:Standalone 下 search 回的 distance 即相似度(越大越像),阈值直接比 score。
BM25 的 distance 是 BM25 分;hybrid 的 distance 是 RRF 融合分。三种 score 不可跨路比较。
"""
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024
MAX_TEXT = 16384
MAX_QUESTION = 2048
MAX_ANSWER = 8192
MAX_SECTION = 512
MAX_CONTENT_TYPE = 32
MAX_CATEGORY = 255

_CLIENT: MilvusClient | None = None

_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Standalone);传 uri 返回独立客户端(测试/临时)。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等建集合:dense + text(chinese analyzer)+ sparse(BM25 Function)+ 标量字段。
    已存在则只确保 load。"""
    if client.has_collection(collection):
        client.load_collection(collection)
        return
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=MAX_TEXT,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("question", DataType.VARCHAR, max_length=MAX_QUESTION)
    schema.add_field("answer", DataType.VARCHAR, max_length=MAX_ANSWER)
    schema.add_field("section_path", DataType.VARCHAR, max_length=MAX_SECTION)
    schema.add_field("content_type", DataType.VARCHAR, max_length=MAX_CONTENT_TYPE)
    schema.add_field("category", DataType.VARCHAR, max_length=MAX_CATEGORY)
    schema.add_function(Function(
        name="bm25_fn", input_field_names=["text"],
        output_field_names="sparse", function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX",
                           metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                           metric_type="BM25",
                           params={"bm25_k1": 1.2, "bm25_b": 0.75})
    client.create_collection(collection, schema=schema, index_params=index_params)
    client.load_collection(collection)


def _fit(text: str, limit: int) -> str:
    """按 UTF-8 字节数裁剪:Milvus VARCHAR max_length 计字节(中文一字 3 字节)。"""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore")


def upsert_vectors(client: MilvusClient, rows: list[dict],
                   collection: str = COLLECTION) -> None:
    """按主键 upsert。所有 VARCHAR 字段先按字节裁剪,免得整批被 Milvus 拒。"""
    if not rows:
        return
    fitted = []
    for r in rows:
        fitted.append({
            "id": r["id"],
            "dense": r["dense"],
            "text": _fit(r["text"], MAX_TEXT),
            "question": _fit(r["question"], MAX_QUESTION),
            "answer": _fit(r["answer"], MAX_ANSWER),
            "section_path": _fit(r.get("section_path") or "", MAX_SECTION),
            "content_type": _fit(r.get("content_type") or "", MAX_CONTENT_TYPE),
            "category": _fit(r.get("category") or "", MAX_CATEGORY),
        })
    client.upsert(collection, fitted)


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:新 upsert 的数据须 flush 后才稳定可被 BM25/检索命中。"""
    client.flush(collection)


def _category_expr(category: str | None) -> str:
    if not category:
        return ""
    safe = category.replace("\\", "\\\\").replace('"', '\\"')
    return f'category == "{safe}"'


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {
        "id": h["id"],
        "score": float(h["distance"]),
        "question": e["question"],
        "answer": e["answer"],
        "section_path": e["section_path"],
        "content_type": e["content_type"],
        "category": e["category"],
    }


def dense_search(client: MilvusClient, vector: list[float], top_k: int,
                 category: str | None = None,
                 collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[vector], anns_field="dense", limit=top_k,
        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
        filter=_category_expr(category),
    )
    return [_hit(h) for h in res[0]]


def bm25_search(client: MilvusClient, text: str, top_k: int,
                category: str | None = None,
                collection: str = COLLECTION) -> list[dict]:
    res = client.search(
        collection, data=[text], anns_field="sparse", limit=top_k,
        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
        filter=_category_expr(category),
    )
    return [_hit(h) for h in res[0]]


def hybrid_search(client: MilvusClient, vector: list[float], text: str, top_k: int,
                  recall: int | None = None, category: str | None = None,
                  collection: str = COLLECTION) -> list[dict]:
    """dense + BM25 各召回 recall 条,RRF(k=60)融合后取 top_k。"""
    recall = recall or settings.recall_top_k
    expr = _category_expr(category)
    dense_req = AnnSearchRequest(
        data=[vector], anns_field="dense",
        param={"metric_type": "COSINE"}, limit=recall, expr=expr,
    )
    bm25_req = AnnSearchRequest(
        data=[text], anns_field="sparse",
        param={"metric_type": "BM25"}, limit=recall, expr=expr,
    )
    res = client.hybrid_search(
        collection, [dense_req, bm25_req], RRFRanker(k=60),
        limit=top_k, output_fields=_OUTPUT,
    )
    return [_hit(h) for h in res[0]]


def count(client: MilvusClient, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0",
                        output_fields=["count(*)"])[0]["count(*)"]


def drop(client: MilvusClient, collection: str = COLLECTION) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)

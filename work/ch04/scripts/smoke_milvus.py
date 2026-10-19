"""冒烟:连 Milvus Standalone,建集合 → upsert → search → count → drop。
独立脚本(不 import app.kb.milvus_client),验证 Standalone 这条链通不通。
需先 make milvus-up(或 docker compose up -d etcd minio milvus-standalone)。
运行:PYTHONPATH=. uv run python scripts/smoke_milvus.py"""
from pymilvus import DataType, MilvusClient

from app.config import settings

COLLECTION = "smoke_knowledge"
DIM = 1024


def main() -> None:
    client = MilvusClient(uri=settings.milvus_uri)
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("question", DataType.VARCHAR, max_length=2048)
    schema.add_field("answer", DataType.VARCHAR, max_length=8192)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    client.load_collection(COLLECTION)

    v = [0.0] * DIM
    v[0] = 1.0
    client.upsert(COLLECTION, [{"id": 1, "vector": v, "question": "运费怎么算", "answer": "满99包邮"}])
    client.flush(COLLECTION)
    res = client.search(COLLECTION, data=[v], anns_field="vector", limit=1,
                        output_fields=["question", "answer"],
                        search_params={"metric_type": "COSINE"})
    assert res[0] and res[0][0]["id"] == 1, f"检索异常:{res}"
    cnt = client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]
    assert cnt == 1, f"计数异常:{cnt}"
    client.drop_collection(COLLECTION)
    print("✅ 冒烟通过:Milvus Standalone 建集合/upsert/search/count 正常")


if __name__ == "__main__":
    main()

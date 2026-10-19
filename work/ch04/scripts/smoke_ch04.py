"""ch04 fail-fast 冒烟:Milvus 原生 BM25 + chinese analyzer + hybrid_search;
本地 bge-reranker-v2-m3 加载 + compute_score(normalize=True)+ CPU 50 对耗时。

运行(在 work/ch04 下,用 ch03 的 venv;首次下载模型时把 HF_HUB_OFFLINE 设为 0):
  HF_HOME='E:\\model' HF_HUB_OFFLINE=1 uv run --project ../ch03 python scripts/smoke_ch04.py

GO/NO-GO:任一断言失败即停,不要继续后面的任务;只在固定选型内换法,并记录到 dev-notes。
"""
import asyncio
import os
import time
import uuid

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

URI = os.environ.get("MILVUS_URI", "http://localhost:19530")


def smoke_milvus() -> None:
    import pymilvus

    print(f"pymilvus {pymilvus.__version__}")
    client = MilvusClient(uri=URI)
    coll = f"smoke_ch04_{uuid.uuid4().hex[:8]}"
    try:
        try:
            print(f"milvus server {client.get_server_version()}")
        except Exception:
            pass
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=8)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=2048,
            enable_analyzer=True, analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
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
        client.create_collection(coll, schema=schema, index_params=index_params)
        client.insert(coll, [
            {"id": 1, "dense": [1.0, 0, 0, 0, 0, 0, 0, 0],
             "text": "智能猫砂盆 MH-LP100 废砂盒每 5 至 7 天倾倒一次"},
            {"id": 2, "dense": [0, 1.0, 0, 0, 0, 0, 0, 0],
             "text": "自动饮水机 MH-W40 滤芯每 3 周更换一次"},
            {"id": 3, "dense": [0, 0, 1.0, 0, 0, 0, 0, 0],
             "text": "满 99 元包邮,未满收取 10 元运费"},
        ])
        client.flush(coll)
        client.load_collection(coll)

        res = client.search(
            coll, data=["MH-LP100 废砂盒"], anns_field="sparse", limit=3,
            output_fields=["text"], search_params={"metric_type": "BM25"},
        )
        ids = [h["id"] for h in res[0]]
        print("BM25:", [(h["id"], round(float(h["distance"]), 4)) for h in res[0]])
        assert ids and ids[0] == 1, f"BM25 型号词未命中: {ids}"

        dense_req = AnnSearchRequest(
            data=[[1.0, 0, 0, 0, 0, 0, 0, 0]], anns_field="dense",
            param={"metric_type": "COSINE"}, limit=3,
        )
        bm25_req = AnnSearchRequest(
            data=["MH-W40 滤芯"], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=3,
        )
        hybrid = client.hybrid_search(
            coll, [dense_req, bm25_req], RRFRanker(k=60), limit=3,
            output_fields=["text"],
        )
        print("hybrid:", [(h["id"], round(float(h["distance"]), 4)) for h in hybrid[0]])
        assert hybrid and len(hybrid[0]) >= 1, "hybrid_search 无结果"
        print("GO: Milvus BM25 + chinese analyzer + hybrid_search 通")
    finally:
        if client.has_collection(coll):
            client.drop_collection(coll)


def smoke_reranker() -> float:
    from FlagEmbedding import FlagReranker

    model_name = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    use_fp16 = os.environ.get("RERANK_DEVICE", "cpu") != "cpu"
    print(f"HF_HOME={os.environ.get('HF_HOME')} HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    t0 = time.perf_counter()
    reranker = FlagReranker(model_name, use_fp16=use_fp16)
    load_s = time.perf_counter() - t0

    query = "自动饮水机 MH-W40 的滤芯多久换一次"
    docs = [
        "MH-W40 建议每 3 周更换一次滤芯",
        "MH-W20 建议每 2 周更换一次滤芯",
        "满 99 元包邮,未满收取 10 元运费",
    ] + [f"无关文本 {i}" for i in range(47)]
    pairs = [[query, d] for d in docs]
    t0 = time.perf_counter()
    scores = reranker.compute_score(pairs, normalize=True)
    elapsed = time.perf_counter() - t0
    if not isinstance(scores, list):
        scores = [scores]
    assert len(scores) == 50, f"应返回 50 个分,实际 {len(scores)}"
    assert all(0.0 <= float(s) <= 1.0 for s in scores), "normalize=True 应落在 0~1"
    assert scores[0] > scores[1] > scores[2], f"相关度顺序不对: {scores[:3]}"
    print(f"GO: reranker 加载 {load_s:.1f}s;50 对 CPU 打分 {elapsed:.1f}s;"
          f"top3={[round(float(s), 4) for s in scores[:3]]}")
    return elapsed


async def main() -> None:
    await asyncio.to_thread(smoke_milvus)
    elapsed = await asyncio.to_thread(smoke_reranker)
    if elapsed > 60:
        print(f"WARN: 50 对重排耗时 {elapsed:.1f}s,评估全量会明显变慢;先用 --limit 迭代")
    print("GO: 两条链路都通")


if __name__ == "__main__":
    asyncio.run(main())

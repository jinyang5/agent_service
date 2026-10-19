"""冒烟:进程内加载 BGE-M3,验证能加载、出向量、维度 1024。go/no-go 风险闸。
首次会从 HuggingFace 下 ~2GB;国内慢/被墙先 export HF_ENDPOINT=https://hf-mirror.com
本脚本刻意直接调 FlagEmbedding(不 import app.core.embeddings),作为写封装之前的独立闸门。
运行:PYTHONPATH=. uv run python scripts/smoke_embed.py"""
from FlagEmbedding import BGEM3FlagModel

from app.config import settings


def main() -> None:
    # CPU 冒烟:不传 use_fp16 / devices
    model = BGEM3FlagModel(settings.embed_model)
    out = model.encode(["邮费是多少", "运费怎么算"], batch_size=12, max_length=8192)
    vecs = out["dense_vecs"]
    dims = [len(v) for v in vecs]
    print(f"返回 {len(vecs)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
    print("✅ 冒烟通过:进程内 BGE-M3,维度 1024")


if __name__ == "__main__":
    main()

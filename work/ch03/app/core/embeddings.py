"""嵌入:进程内加载 BGE-M3(dense, dim 1024),本地推理,不走网络、不需要 key。

模型只在首次调用时加载(惰性单例);CPU 推理是阻塞的,丢到线程跑,别卡住 async 事件循环。
GPU 时用 fp16 + 指定 devices;CPU 时两者都省(FlagEmbedding 会报未知参数或走错路径)。
"""
import asyncio

from FlagEmbedding import BGEM3FlagModel

from app.config import settings

_MODEL = None


def _model() -> BGEM3FlagModel:
    """惰性单例。供测试 monkeypatch(_model 返回带 .encode 的假对象)。"""
    global _MODEL
    if _MODEL is None:
        kwargs: dict = {}
        if settings.embed_device and settings.embed_device != "cpu":
            kwargs = {"use_fp16": True, "devices": [settings.embed_device]}
        _MODEL = BGEM3FlagModel(settings.embed_model, **kwargs)
    return _MODEL


def _encode(texts: list[str]) -> list[list[float]]:
    out = _model().encode(texts, batch_size=12, max_length=8192)
    return [list(map(float, v)) for v in out["dense_vecs"]]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_encode, texts)


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]

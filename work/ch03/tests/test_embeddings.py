from app.core import embeddings


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=None, max_length=None):
        self.calls.append((list(texts), batch_size, max_length))
        return {"dense_vecs": [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)]}


async def test_embed_texts_uses_model_and_dense_vecs(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "_model", lambda: fake)
    out = await embeddings.embed_texts(["a", "b"])
    assert out == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert fake.calls == [(["a", "b"], 12, 8192)]


async def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", lambda: _FakeModel())
    v = await embeddings.embed_query("邮费")
    assert v == [0.0, 0.0, 1.0]

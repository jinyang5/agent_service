import pytest
from pydantic import ValidationError

from app.config import Settings


def test_account_fields_required(monkeypatch):
    # 账号三件套不给默认值:给了等于替人猜,猜错不在启动时报错,而是第一次调模型时抛看不懂的 401
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "m")
    monkeypatch.setenv("CHAT_BASE_URL", "u")
    monkeypatch.setenv("CHAT_API_KEY", "k")
    monkeypatch.setenv("TOKEN_BUDGET", "500")
    assert Settings(_env_file=None).token_budget == 500


def test_ch04_settings_defaults(monkeypatch):
    # 默认值断言必须隔离外部环境变量:本机 shell 里一个 RECALL_TOP_K=999
    # 就会让「默认 50」的断言误报,跟代码是否漂移无关。
    for k in ("TEST_DATABASE_URL", "RERANK_MODEL", "RERANK_DEVICE",
              "RECALL_TOP_K", "RERANK_TOP_K", "RERANK_MIN_SCORE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.test_database_url.endswith("/mewhelp_ch04_test")
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_device == "cpu"
    assert s.recall_top_k == 50
    assert s.rerank_top_k == 10
    assert s.rerank_min_score == 0.3

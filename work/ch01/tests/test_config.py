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

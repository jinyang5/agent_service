from app.config import settings
from app.core.llm import get_chat_model


def test_factory_points_to_chat_upstream():
    m = get_chat_model()
    assert m.model_name == settings.chat_model       # 跟随配置,不硬编码模型名
    assert str(m.openai_api_base) == settings.chat_base_url


def test_factory_streaming_flag():
    assert get_chat_model(streaming=True).streaming is True

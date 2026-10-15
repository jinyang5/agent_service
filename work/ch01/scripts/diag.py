"""诊断:直接调一次上游,把真实报错打出来(不打印 key 任何内容)。"""
from app.config import settings
from app.core.llm import get_chat_model

print("model =", settings.chat_model)
print("base_url =", settings.chat_base_url)
print("key 已加载 =", bool(settings.chat_api_key.strip()), "| 格式像 sk- =", settings.chat_api_key.startswith("sk-"), "| 长度>20 =", len(settings.chat_api_key) > 20)

model = get_chat_model()
try:
    resp = model.invoke("你好")
    print("上游 OK,回复:", resp.content)
except Exception:
    import traceback
    traceback.print_exc()

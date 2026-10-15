r"""离线看懂 with_structured_output 的完整数据流(不用 key、不联网)。

跑法(在 work/ch01 目录下):
    .\.venv\Scripts\python.exe -m scripts.explore_structured_output
"""
import json

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _oai_structured_outputs_parser

from app.schemas.extract import AfterSalesTicket, RequestType

LINE = "=" * 64


def show(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


# ---------- ① 源头:你自己写的枚举 ----------
show("① 源头:你键盘敲出来的枚举(schemas/extract.py)")
for m in RequestType:
    print(f"   {m.name:10s} = {m.value!r}")

# ---------- ② pydantic 从类翻译出的 schema ----------
show("② pydantic 自动翻译出的 JSON Schema(字段清单 + enum)")
print(json.dumps(AfterSalesTicket.model_json_schema(), ensure_ascii=False, indent=2))

# ---------- ③ 真正发出去的 HTTP 报文(用假网络拦下来看) ----------
show("③ 真正发给 DeepSeek 的请求体(用假网络拦截,真流程真报文)")

captured: dict = {}

FAKE_REPLY = {
    "id": "chatcmpl-fake", "object": "chat.completion",
    "created": 0, "model": "fake",
    "choices": [{
        "index": 0, "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": '{"order_id":"MH20260701123","request_type":"退款","expected_solution":"到货损坏要求退款"}',
            "parsed": {"order_id": "MH20260701123", "request_type": "退款",
                       "expected_solution": "到货损坏要求退款"},
        },
    }],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def fake_upstream(request: httpx.Request) -> httpx.Response:
    captured["body"] = json.loads(request.content)
    return httpx.Response(200, json=FAKE_REPLY)


model = ChatOpenAI(
    model="deepseek-v4-flash", base_url="http://fake.local", api_key="sk-fake",
    http_client=httpx.Client(transport=httpx.MockTransport(fake_upstream)),
)
extractor = model.with_structured_output(AfterSalesTicket)
result = extractor.invoke("订单 MH20260701123 的猫爬架散架了,我要退款")

print("--- 请求体里的 response_format(全部由 LangChain 自动加的)---")
print(json.dumps(captured["body"]["response_format"], ensure_ascii=False, indent=2))

# ---------- ④ 模型回复 → 解析成对象 ----------
show("④ 模型返回 JSON 后,解析成你的对象")
print("   合法回复 →", result)

show("⑤ 你写的兜底:模型回占位串 'null' 时怎么处理")
占位 = AIMessage("", additional_kwargs={"parsed": {
    "order_id": "null", "request_type": "退款", "expected_solution": "退款"}})
print("   原始 parsed:", {"order_id": "null"})
print("   归一后    :", _oai_structured_outputs_parser(占位, schema=AfterSalesTicket))

show("⑥ 枚举外的值会被拦下")
try:
    _oai_structured_outputs_parser(
        AIMessage("", additional_kwargs={"parsed": {
            "order_id": None, "request_type": "退货", "expected_solution": "退货"}}),
        schema=AfterSalesTicket,
    )
    print("   ??? 居然通过了(不应该)")
except Exception as exc:
    print(f"   报错类型: {type(exc).__name__}")
    print(f"   报错内容: {str(exc).splitlines()[0]}")

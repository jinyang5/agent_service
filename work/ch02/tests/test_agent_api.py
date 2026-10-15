from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

from app.core import agent
from app.core.agent import AgentResult
from app.main import app
from app.tools.infra import ToolRun


def _fake_result():
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    return AgentResult(
        conversation_id=12,
        answer="订单 1001 正在运输中。",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        tool_runs=[ToolRun("c1", "query_logistics", True, tm)],
    )


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id, model=None):
        return _fake_result()
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise agent.ConversationNotFound(999)
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404


def test_agent_endpoint_422_on_missing_fields():
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422


def test_agent_endpoint_422_on_blank_message():
    # 请求模型经 FastAPI 校验:纯空白消息同样 422
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "   "})
    assert r.status_code == 422


def test_agent_endpoint_503_on_db_error(monkeypatch):
    # Review Focus 5:repository 抛 SQLAlchemyError 应分诊到 503(而非 502)
    from sqlalchemy.exc import OperationalError
    async def fake_run(*a, **k):
        raise OperationalError("boom", None, Exception("boom"))
    monkeypatch.setattr(agent, "run_agent_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503

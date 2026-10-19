from langchain_core.messages import AIMessage, HumanMessage

from app.core.memory import SessionStore, trim_history


def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []


def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("嗨"))
    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1


def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "喵" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]


def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs


def test_trim_keeps_single_over_budget_message():
    huge = HumanMessage("喵" * 10000)
    assert trim_history([huge], max_tokens=100) == [huge]


def test_trim_drops_oldest_but_keeps_current_when_over_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("喵" * 10000)]
    assert trim_history(msgs, max_tokens=100) == [msgs[-1]]

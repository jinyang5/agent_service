from langchain_core.messages import HumanMessage

from app.core.prompts import (
    AGENT_PROMPT,
    AGENT_SYSTEM,
    CUSTOMER_SERVICE_PROMPT,
    EXTRACT_PROMPT,
    RAG_ANSWER_PROMPT,
    SELF_CHECK_PROMPT,
)


def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(history=[HumanMessage("你们卖猫粮吗?")])
    assert msgs[0].type == "system"
    assert msgs[-1].content == "你们卖猫粮吗?"
    # 行为约束必须写进 system prompt
    for keyword in ("客服", "不", "订单"):
        assert keyword in msgs[0].content


def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="订单 MH1 坏了要退款")
    assert msgs[0].type == "system"
    assert "订单 MH1 坏了要退款" in msgs[-1].content


def test_agent_system_covers_tool_principles():
    for kw in ["小喵", "工具", "query_faq", "create_ticket", "不要臆造"]:
        assert kw in AGENT_SYSTEM


def test_agent_prompt_has_history_placeholder():
    assert any(
        getattr(m, "variable_name", None) == "history" for m in AGENT_PROMPT.messages
    )


def test_rag_answer_prompt_has_citations_and_negative_knowledge():
    from app.core.prompts import RAG_ANSWER_SYSTEM

    for kw in ("[n]", "[1][2]", "证据", "以平台售后规则为准", "不要编造"):
        assert kw in RAG_ANSWER_SYSTEM


def test_evidence_refusal_is_fixed():
    from app.core.prompts import EVIDENCE_REFUSAL

    assert "没有查到" in EVIDENCE_REFUSAL
    assert "人工客服" in EVIDENCE_REFUSAL


def test_rag_answer_prompt_renders_query_and_numbered_evidence():
    msgs = RAG_ANSWER_PROMPT.format_messages(
        query="满多少包邮?", evidence="[1] 满 99 元包邮"
    )
    assert msgs[0].type == "system"
    assert "满多少包邮?" in msgs[-1].content
    assert "[1] 满 99 元包邮" in msgs[-1].content


def test_rag_answer_system_blocks_timing_and_compensation_promises():
    """负面知识要覆盖到账/到货/维修时效与赔偿,并统一到平台规则。"""
    from app.core.prompts import RAG_ANSWER_SYSTEM

    for kw in ("到账", "到货", "维修", "赔偿", "以平台实际处理为准", "人工客服"):
        assert kw in RAG_ANSWER_SYSTEM


def test_self_check_prompt_renders_query_and_evidence():
    msgs = SELF_CHECK_PROMPT.format_messages(query="q", evidence="[1] e")
    assert msgs[0].type == "system"
    assert "证据" in msgs[0].content
    assert "q" in msgs[-1].content
    assert "[1] e" in msgs[-1].content

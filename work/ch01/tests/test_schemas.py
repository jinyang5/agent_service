import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.extract import AfterSalesTicket, ExtractRequest, RequestType


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_extract_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        ExtractRequest(text="")


def test_ticket_order_id_nullable():
    t = AfterSalesTicket(order_id=None, request_type="退款", expected_solution="退全款")
    assert t.order_id is None
    assert t.request_type is RequestType.REFUND


def test_ticket_normalizes_placeholder_order_id():
    # 模型常把「没有订单号」表达成占位串而非省略参数,归一为 None
    t = AfterSalesTicket(order_id="null", request_type="退款", expected_solution="退全款")
    assert t.order_id is None


def test_ticket_rejects_unknown_request_type():
    with pytest.raises(ValidationError):
        AfterSalesTicket(order_id=None, request_type="砍价", expected_solution="x")

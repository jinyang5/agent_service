import logging
import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core import citations, query_understanding, retrieval, selfcheck
from app.db import repository

logger = logging.getLogger(__name__)


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")


class LogisticsInput(BaseModel):
    order_id: str = Field(description="订单号,用于查询该订单的物流轨迹")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间和商品名。用于用户询问某个订单情况时。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
    }


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }


@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """查询订单的物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。"""
    rng = random.Random(f"logistics:{order_id}")
    status = rng.choice(["已揽件", "运输中", "派送中", "已签收"])
    city = rng.choice(["深圳", "广州", "杭州", "上海", "成都"])
    return {
        "order_id": order_id,
        "status": status,
        "location": f"{city}分拨中心",
        "timeline": [f"{city}分拨中心 已发出", f"当前状态:{status}"],
    }


class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作/商品规格类问题(可用原话)")
    category: str | None = Field(
        default=None,
        description="可选:按 category 过滤。一般留空;只有用户明确说品类时才填,如『商品规格手册』『退货退款政策』",
    )


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策/商品规格知识库(混合检索 + 重排)。

    返回带 [n] 编号的证据供作答引用。证据不足时返回 sufficient=False 与
    signal=evidence_insufficient,调用方必须直接拒答,不要用自身知识编造。
    """
    try:
        u = await query_understanding.understand(keyword)
    except Exception as exc:  # noqa: BLE001 - 改写失败必须回退原话
        logger.warning("Query 改写失败(%s),回退原话", type(exc).__name__)
        u = {"standard": keyword, "expanded": []}

    standard = u["standard"] or keyword
    bm25_text = standard + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        standard, strategy="hybrid_rerank", category=category, bm25_text=bm25_text)

    # ⑤ 机械闸:无命中 or 最高 rerank 分低于阈值 → 拒答,不落池
    top_score = float(hits[0].get("rerank_score", 0.0)) if hits else 0.0
    if not hits or top_score < settings.rerank_min_score:
        return {
            "sufficient": False,
            "signal": "evidence_insufficient",
            "source": "retrieval_low_conf",
            "reason": f"检索证据不足(top={top_score:.3f})",
            "citations": [],
        }

    # ⑥ 自评闸:生成前判断证据够不够;只有 useful=false 才落池(由 agent 写)
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    try:
        chk = await selfcheck.check_sufficient(standard, ev_texts)
    except Exception as exc:  # noqa: BLE001 - 自评异常须 fail-open,不能当拒答
        # 自评上游异常不是「judged false」:fail-open 继续,避免误把异常当拒答/入池
        logger.warning("证据自评失败(%s),按证据可用继续", type(exc).__name__)
        chk = {"useful": True, "reason": "self_check_unavailable"}
    if not chk.get("useful", True):
        return {
            "sufficient": False,
            "signal": "evidence_insufficient",
            "source": "self_check",
            "reason": chk.get("reason", ""),
            "citations": [],
        }

    # ⑦ 首尾组装 + ⑧ 编号证据
    arranged = retrieval.arrange_head_tail(hits)
    cites = citations.build_citations(arranged)
    return {
        "sufficient": True,
        "evidence": citations.build_evidence(cites),
        "citations": cites,
    }


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """当用户问题需要人工介入(投诉、无法自助解决、明确要求人工)时,创建人工工单。
    description 填用户问题描述,ticket_type 从 售后/投诉/咨询 中选。"""
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}

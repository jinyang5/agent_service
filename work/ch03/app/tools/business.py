import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.core import retrieval
from app.db import repository


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
    keyword: str = Field(description="用于在常见问题库中检索的关键词,如『退货』『发货时效』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str) -> dict:
    """根据关键词查询常见问题解答(FAQ)。用于用户咨询政策、规则、操作流程等通用问题时。"""
    hits = await retrieval.search_knowledge(keyword)
    if not hits:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": [{"question": h["question"], "answer": h["answer"]} for h in hits]}


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

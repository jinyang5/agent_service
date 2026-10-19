from app.tools.business import query_logistics, query_order, query_product


async def test_query_order_deterministic_and_shaped():
    r1 = await query_order.ainvoke({"order_id": "1001"})
    r2 = await query_order.ainvoke({"order_id": "1001"})
    assert r1 == r2                                  # 同种子可复现
    assert r1["order_id"] == "1001"
    assert r1["status"] in {"待付款", "已付款", "已发货", "已签收"}


async def test_query_product_and_logistics_names():
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
    p = await query_product.ainvoke({"product_name": "猫粮"})
    assert p["product_name"] == "猫粮" and "price" in p
    lg = await query_logistics.ainvoke({"order_id": "1001"})
    assert lg["order_id"] == "1001" and "status" in lg and "timeline" in lg

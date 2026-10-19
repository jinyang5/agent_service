"""验收:两轮流式对话,第二轮须接住第一轮上下文(人工看第二轮是否复述姓名/商品)。"""
import json

import httpx

BASE = "http://localhost:8000"
SESSION_ID = "demo-ctx"


def ask(message: str) -> None:
    print(f"\n>>> 用户: {message}")
    print("客服(流式): ", end="", flush=True)
    with httpx.stream(
        "POST", f"{BASE}/api/chat",
        json={"session_id": SESSION_ID, "message": message}, timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                print(json.loads(line[len("data: "):])["delta"], end="", flush=True)
    print()


if __name__ == "__main__":
    ask("我叫王小明,昨天买了你们的智能猫砂盆")
    ask("还记得我叫什么、买了什么吗?")

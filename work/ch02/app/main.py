from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MewHelp Ch02", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)


@app.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    """网页聊天界面(同源提供,前端直接 POST /api/chat)。"""
    return FileResponse(STATIC_DIR / "index.html")

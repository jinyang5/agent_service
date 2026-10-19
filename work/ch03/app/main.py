from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MewHelp Ch02", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(agent_router)
app.include_router(kb_router)
app.include_router(admin_router)
app.include_router(jobs_router)

# 后台各页共用外壳(导航样式 + mountAdminNav),抽成文件放静态目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    """网页聊天界面(同源提供,前端直接 POST /api/chat)。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/kb", include_in_schema=False)
async def kb_page() -> FileResponse:
    """ch03 知识库录入页:贴文档就能切块入库、向量化、当场检索自测。"""
    return FileResponse(STATIC_DIR / "kb.html")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    """后台管理首页:知识库一张卡,一处进出;各模块页面保持各自原路径。"""
    return FileResponse(STATIC_DIR / "admin.html")

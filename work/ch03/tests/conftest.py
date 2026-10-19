# tests/conftest.py
import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL_FILES = [
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch03-ddl.sql",
]
# 删除顺序:先子表后父表;knowledge_chunks 自引用 FK 靠 FOREIGN_KEY_CHECKS=0 兜
_TABLES = ["messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]
_DB_NAME = settings.test_database_url.rsplit("/", 1)[-1]


def _ddl_statements() -> list[str]:
    """把权威 DDL 拆成可逐句执行的语句。

    先剥掉 `--` 行注释再按 `;` 切:DDL 里注释本身含中文分号,直接 `split(";")`
    会切坏语句;且每条 CREATE TABLE 前都有一行注释,不剥注释则过滤会漏掉建表句。
    """
    stmts: list[str] = []
    for ddl in _DDL_FILES:
        lines = [ln for ln in ddl.read_text(encoding="utf-8").splitlines()
                 if not ln.lstrip().startswith("--")]
        stmts += [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    return stmts


@pytest_asyncio.fixture(scope="session")
async def _test_engine():
    # 用 server-url(不带库名)建测试库,库名取 settings.test_database_url(mewhelp_ch03_test)
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {_DB_NAME}"))
        await conn.execute(text(f"CREATE DATABASE {_DB_NAME} CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(权威源:sql/ch02-ddl.sql + sql/ch03-ddl.sql;按 ; 分割逐句执行)
    async with engine.begin() as conn:
        for s in _ddl_statements():
            if s.lstrip().upper().startswith("CREATE TABLE"):
                await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, db_clean, monkeypatch):
    """把 repository 用到的 async_session 指向测试库;跟着 db_clean 走,每个用例跑在干净的表上。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory


@pytest_asyncio.fixture()
async def db_clean(_test_engine):
    """每个测试后清空全部测试表(ch02 四张 + ch03 两张;关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))


@pytest_asyncio.fixture()
async def client():
    from httpx import ASGITransport, AsyncClient
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

# tests/conftest.py
import pathlib

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings

_DDL = pathlib.Path(__file__).resolve().parents[3] / "sql" / "ch02-ddl.sql"
_TABLES = ["messages", "tickets", "conversations", "faq"]  # 删除顺序:先子表后父表


def _ddl_statements() -> list[str]:
    """把权威 DDL 拆成可逐句执行的语句。

    先剥掉 `--` 行注释再按 `;` 切:DDL 里 FAQ 那句注释本身含中文分号
    (`数据源;ch03 起…`),直接 `split(";")` 会把语句切坏;且每条 CREATE TABLE
    前都有一行注释,不剥注释则 `startswith("CREATE TABLE")` 过滤会漏掉全部建表句。
    """
    lines = [
        ln for ln in _DDL.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("--")
    ]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


@pytest_asyncio.fixture(scope="session")
async def _test_engine():
    # 用 server-url(不带库名)建 mewhelp_ch02_test 库
    server_url = settings.test_database_url.rsplit("/", 1)[0]
    admin = create_async_engine(server_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text("DROP DATABASE IF EXISTS mewhelp_ch02_test"))
        await conn.execute(text("CREATE DATABASE mewhelp_ch02_test CHARACTER SET utf8mb4"))
    await admin.dispose()

    engine = create_async_engine(settings.test_database_url, pool_pre_ping=True)
    # 跑 DDL 建表(以 ch02-ddl.sql 为权威;按 ; 分割逐句执行)
    async with engine.begin() as conn:
        for s in _ddl_statements():
            if s.lstrip().upper().startswith("CREATE TABLE"):
                await conn.execute(text(s))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def db_session_factory(_test_engine, monkeypatch):
    """把 repository 用到的 async_session 指向测试库。"""
    factory = async_sessionmaker(_test_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.base.async_session", factory)
    return factory


@pytest_asyncio.fixture()
async def db_clean(_test_engine):
    """每个测试后清空 4 张表(关外键检查以便 TRUNCATE)。"""
    yield
    async with _test_engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in _TABLES:
            await conn.execute(text(f"TRUNCATE TABLE {t}"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

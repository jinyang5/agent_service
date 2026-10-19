"""真跑作业运行器:真起进程、真收进程,不 monkeypatch start/stop。

argv 只在测试里换成短活命令(dataclasses.replace 注入,monkeypatch 自动还原);
JOBS 的名字白名单原样不动,生产语义不被削弱。日志落 tmp_path,不进仓库。"""
import asyncio
import dataclasses
import sys
from collections.abc import Callable

import pytest

from app.core import jobs

NAME = "kb-preview"          # 白名单里的名字;argv 由 _inject 换成短活命令


def test_eval_rag_job_registered():
    """RAG 评估页「重跑」按钮指向的作业:argv 写死,分钟级,算重活。"""
    spec = jobs.JOBS["eval-rag"]
    assert spec.argv == ("make", "eval-rag")
    assert spec.heavy is True


def _inject(monkeypatch, tmp_path, code: str) -> None:
    """日志指到 tmp_path;argv 换成 `python -c <code>`,其余字段一字不改。"""
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    monkeypatch.setitem(jobs.JOBS, NAME,
                        dataclasses.replace(jobs.JOBS[NAME], argv=(sys.executable, "-c", code)))


async def _wait_until(name: str, pred: Callable[[dict], bool], timeout: float = 15.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        st = jobs.status(name)
        if pred(st):
            return st
        if loop.time() > deadline:
            raise AssertionError(f"超时:{name} 现在 status={st['status']!r} rc={st['returncode']}")
        await asyncio.sleep(0.05)


async def test_real_run_success_sets_ok_and_logs(tmp_path, monkeypatch):
    _inject(monkeypatch, tmp_path, "print('hello-runner')")
    run = await jobs.start(NAME)
    assert run.status == "running" and run.pid
    st = await _wait_until(NAME, lambda s: s["status"] == "ok")
    assert st["returncode"] == 0
    assert st["started_at"] and st["finished_at"]
    log = jobs.tail(NAME)
    assert log.splitlines()[0] == f"$ {sys.executable} -c print('hello-runner')"
    assert "hello-runner" in log


async def test_real_run_nonzero_rc_sets_failed(tmp_path, monkeypatch):
    _inject(monkeypatch, tmp_path, "import sys; sys.exit(7)")
    await jobs.start(NAME)
    st = await _wait_until(NAME, lambda s: s["status"] == "failed")
    assert st["returncode"] == 7


async def test_reentry_refused_while_process_really_runs(tmp_path, monkeypatch):
    _inject(monkeypatch, tmp_path, "import time; time.sleep(30)")
    await jobs.start(NAME)
    with pytest.raises(RuntimeError, match="正在运行中"):
        await jobs.start(NAME)
    await jobs.stop(NAME)                        # 真停:顺带验证 stop 收得掉活进程
    st = await _wait_until(NAME, lambda s: s["returncode"] is not None)
    assert st["status"] == "stopped"


async def test_concurrent_double_submit_spawns_only_once(tmp_path, monkeypatch):
    """双击「重跑」:两次 start 同时进来,只许一个过守卫,另一个走 409 那条路,日志不许被二次覆盖。"""
    _inject(monkeypatch, tmp_path, "import time; time.sleep(30)")
    results = await asyncio.gather(jobs.start(NAME), jobs.start(NAME), return_exceptions=True)
    oks = [r for r in results if not isinstance(r, Exception)]
    errs = [r for r in results if isinstance(r, Exception)]
    assert len(oks) == 1, f"并发双击起了 {len(oks)} 份进程"
    assert len(errs) == 1 and isinstance(errs[0], RuntimeError)
    assert jobs.log_path(NAME).read_text(encoding="utf-8").count("由后台页发起") == 1
    await jobs.stop(NAME)
    await _wait_until(NAME, lambda s: s["returncode"] is not None)


async def test_stop_failure_rolls_back_status_and_keeps_guard(tmp_path, monkeypatch):
    """信号发不出去时:状态不许提前写成 stopped,重入党卫不许被绕过。"""
    _inject(monkeypatch, tmp_path, "import time; time.sleep(30)")
    await jobs.start(NAME)

    real_kill = jobs._kill_tree

    async def boom(proc, hard=False):
        raise OSError("模拟:信号发不出去")

    monkeypatch.setattr(jobs, "_kill_tree", boom)
    with pytest.raises(OSError, match="信号发不出去"):
        await jobs.stop(NAME)
    assert jobs.status(NAME)["status"] == "running"        # 状态回滚,没被写成 stopped
    with pytest.raises(RuntimeError, match="正在运行中"):   # 旧进程还活着 → 重入照样被拒
        await jobs.start(NAME)

    monkeypatch.setattr(jobs, "_kill_tree", real_kill)     # 收尾:恢复真杀法,把活进程收掉
    await jobs.stop(NAME)
    await _wait_until(NAME, lambda s: s["returncode"] is not None)


async def test_stop_without_running_job_raises(tmp_path, monkeypatch):
    _inject(monkeypatch, tmp_path, "print('unused')")
    with pytest.raises(RuntimeError, match="没有在运行"):
        await jobs.stop(NAME)

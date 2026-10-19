import pathlib

from app.core import jobs

MAKEFILE = pathlib.Path(__file__).resolve().parents[1] / "Makefile"   # tests/ → work/ch03/


async def test_unknown_job_is_rejected(client):
    """作业名不在白名单 → 404;命令片段没有任何地方能传进来。"""
    assert (await client.post("/api/jobs/rm-rf")).status_code == 404
    assert (await client.get("/api/jobs/rm-rf")).status_code == 404


async def test_running_job_refuses_reentry(client, monkeypatch):
    """同名作业还在跑 → 409,别让两份进程抢同一批产物。"""
    async def running(name):
        raise RuntimeError("正在运行中")
    monkeypatch.setattr(jobs, "start", running)
    assert (await client.post("/api/jobs/kb-build")).status_code == 409


async def test_stop_when_not_running_is_409(client):
    """没在跑就没什么可停的:409 说清楚,而不是 500。"""
    assert (await client.post("/api/jobs/kb-build/stop")).status_code == 409


async def test_stop_failure_is_clean_http_error(client, monkeypatch):
    """停止失败(信号发不出去)→ 结构化 HTTP 错误带 detail,不是裸 500。"""
    async def boom(name):
        raise OSError("停止失败:taskkill /T 退出码 1")
    monkeypatch.setattr(jobs, "stop", boom)
    r = await client.post("/api/jobs/kb-build/stop")
    assert r.status_code == 502
    assert "停止失败" in r.json()["detail"]


def test_every_job_target_exists_in_makefile():
    """注册表里每个作业都得是 Makefile 里真有的目标,免得按钮指向不存在的配方。"""
    text = MAKEFILE.read_text(encoding="utf-8")
    for spec in jobs.JOBS.values():
        assert spec.argv[:2] == ("make", spec.argv[1])
        assert f"\n{spec.argv[1]}:" in text, f"{spec.argv[1]} 不在 Makefile"

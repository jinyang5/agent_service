"""后台作业运行器:把 make 目标搬到浏览器上按,日志轮询回显,长活能停。

安全边界:只能跑 JOBS 注册表里的目标,argv 全部写死在本模块。前端只传一个 job 名,
传不进任何命令片段、参数或路径——页面有「重跑」按钮,但没有 shell。
"""
import asyncio
import contextlib
import datetime as dt
import logging
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]      # work/ch03
LOG_DIR = REPO_ROOT / "log" / "acceptance"
TAIL_LINES = 400


@dataclass(frozen=True)
class JobSpec:
    name: str
    title: str
    argv: tuple[str, ...]
    needs: str
    heavy: bool = False


JOBS: dict[str, JobSpec] = {
    j.name: j for j in (
        JobSpec("kb-preview", "材料清单与切块预览", ("make", "kb-preview"), "本地跑,不写库"),
        JobSpec("kb-build", "离线建库(文档切块 → pending)", ("make", "kb-build"), "需 mysql"),
        JobSpec("kb-mine", "对话挖知识(抽 QA → 去重 → pending)", ("make", "kb-mine"),
                "需 mysql + 聊天上游", heavy=True),
        JobSpec("kb-vectorize", "向量化(嵌入 → Milvus → 回标 done)", ("make", "kb-vectorize"),
                "需 mysql + Milvus"),
        JobSpec("seed-conv", "灌历史会话种子", ("make", "seed-conv"), "需 docker mysql 容器"),
        JobSpec("kb-reset", "清库重建(清两表 + drop 集合)", ("make", "kb-reset"),
                "需 mysql + Milvus;会清空知识库", heavy=True),
    )
}


@dataclass
class JobRun:
    status: str = "idle"          # idle | starting | running | ok | failed | stopped
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    watcher: asyncio.Task | None = field(default=None, repr=False)
    stopping: bool = field(default=False, repr=False)   # 终止在途:watcher 别把这次人为 kill 记成「失败」


_runs: dict[str, JobRun] = {}


def _run(name: str) -> JobRun:
    return _runs.setdefault(name, JobRun())


def log_path(name: str) -> Path:
    return LOG_DIR / f"{name}.log"


async def start(name: str) -> JobRun:
    """发起一次运行。同名作业未结束时拒绝重入(RuntimeError)。"""
    spec = JOBS[name]
    run = _run(name)
    if run.status == "starting":
        raise RuntimeError(f"{spec.title}正在启动中,等它跑起来再发起")
    if run.status == "running":
        raise RuntimeError(f"{spec.title}正在运行中(pid {run.pid}),等它跑完再发起")

    # 同步占坑:从检查到占位之间没有 await——双击两次重跑,第二次在这就被拒,
    # 而不是两份进程都拉起来、后一份把前一份的 proc 与日志覆盖掉。
    prev = (run.status, run.pid)
    run.status, run.pid = "starting", None

    fh = None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = log_path(name)
        header = (f"$ {' '.join(spec.argv)}\n"
                  f"# {dt.datetime.now().isoformat(timespec='seconds')} 由后台页发起\n\n")
        path.write_text(header, encoding="utf-8")      # 每次覆盖:窗口里只看本次
        fh = path.open("a", encoding="utf-8", buffering=1)
        proc = await asyncio.create_subprocess_exec(
            *spec.argv, cwd=REPO_ROOT, stdout=fh, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,     # 独立会话:kill 时连 make → uv → python 一起收
        )
    except Exception:
        if fh is not None:
            fh.close()
        run.status, run.pid = prev      # 起不来:状态还原,按钮可以再点
        raise

    run.status, run.pid = "running", proc.pid
    run.started_at = dt.datetime.now().isoformat(timespec="seconds")
    run.finished_at = run.returncode = None
    run.proc = proc
    run.watcher = asyncio.create_task(_watch(name, run, proc, fh))
    return run


async def _watch(name: str, run: JobRun, proc: asyncio.subprocess.Process, fh) -> None:
    try:
        rc = await proc.wait()
    finally:
        with contextlib.suppress(Exception):
            fh.close()
    run.returncode = rc
    run.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    if not run.stopping and run.status != "stopped":   # 人为 kill 的退出码也非 0,别报成「失败」冤枉它
        run.status = "ok" if rc == 0 else "failed"
    run.proc = None


async def _kill_tree(proc: asyncio.subprocess.Process, *, hard: bool = False) -> None:
    """终止整棵进程树。

    POSIX:杀进程组(killpg)——start_new_session 让 make 自成组长,信号连 uv → python 一起收。
    Windows:没有进程组、start_new_session 也是空操作,只能请 taskkill /T 把子孙一起收掉(/F 强杀)。
    """
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(proc.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await killer.communicate()
        if killer.returncode not in (0, 128):      # 128 = 进程已经不在了,等价于「已经停了」
            raise OSError(f"taskkill /T 退出码 {killer.returncode}:"
                          f"{err.decode('gbk', 'replace').strip()}")
        return
    with contextlib.suppress(ProcessLookupError):  # 进程组已不在 = 已经停了
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL if hard else signal.SIGTERM)


async def stop(name: str) -> None:
    """停一次运行:先发信号,进程真收掉了才写 stopped。

    顺序不能反:状态先写 stopped 而信号其实发不出去(如 Windows 上 os.killpg 不存在),
    重入守卫会以为作业已结束,下一次 start 就把旧进程晾在一边——正是要防的事。
    """
    run = _run(name)
    proc = run.proc
    if run.status != "running" or proc is None:
        raise RuntimeError("该作业当前没有在运行")
    prev_status = run.status
    run.stopping = True         # 同步置位:kill 窗口内 watcher 不把这次人为终止记成「失败」
    try:
        await _kill_tree(proc)                          # 1) 先发信号(POSIX 进程组 / Windows taskkill /T)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except TimeoutError:                            # 2) 软杀收不走:强杀再等
            await _kill_tree(proc, hard=True)
            await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception as e:
        run.stopping = False
        run.status = prev_status                        # 3) 没停成:状态回滚,在跑就还是「running」
        raise OSError(f"停止失败:{type(e).__name__}: {e}") from e
    run.stopping = False
    run.status = "stopped"                              # 4) 确认收掉了,才落终态


def tail(name: str, lines: int = TAIL_LINES) -> str:
    path = log_path(name)
    if not path.exists():
        return ""
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(rows[-lines:])


def status(name: str, with_log: bool = False) -> dict:
    spec, run = JOBS[name], _run(name)
    path = log_path(name)
    out = {
        "name": name, "title": spec.title, "cmd": " ".join(spec.argv),
        "needs": spec.needs, "heavy": spec.heavy,
        "status": run.status, "pid": run.pid,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "returncode": run.returncode,
        "log_mtime": (dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                      if path.exists() else None),
    }
    if with_log:
        out["log"] = tail(name)
    return out


def status_all() -> list[dict]:
    return [status(n) for n in JOBS]

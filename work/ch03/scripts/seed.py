"""把 sql/ch02-seed.sql 灌进容器的 mewhelp 库(幂等:seed.sql 自带 DELETE 后 INSERT)。"""
import subprocess
import sys
from pathlib import Path

SEED = Path(__file__).resolve().parents[3] / "sql" / "ch02-seed.sql"


def main() -> None:
    sql = SEED.read_text(encoding="utf-8")
    r = subprocess.run(
        ["docker", "compose", "exec", "-T", "mysql",
         "mysql", "-uroot", "-proot", "--default-character-set=utf8mb4", "mewhelp"],
        input=sql.encode("utf-8"), capture_output=True,
    )
    sys.stdout.write(r.stdout.decode("utf-8", "replace"))
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()

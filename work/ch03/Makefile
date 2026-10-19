# Python 强制 UTF-8(中文 Windows 默认 GBK,读带中文的 markdown/jsonl 会 UnicodeDecodeError)
export PYTHONUTF8 = 1

.PHONY: dev milvus-up milvus-down kb-preview kb-build kb-vectorize kb-mine seed-conv kb-reset eval-retrieval eval-mining test

# 起点。Windows 上两条坑都踩不得(都会让 /kb 作业按钮全灭):
# 1) uv + CPython 3.14 venv 转发器:进程树 Low integrity,再起 make(Git sh)0xC0000022;
# 2) --reload:uvicorn 在 Windows 配 SelectorEventLoop,而它不支持 asyncio 子进程。
# 所以绕开转发器、不带 --reload,用 uv 基础解释器 + venv 的 site-packages;POSIX 照旧 uv run。
dev:
	if [ -d .venv/Lib/site-packages ]; then \
	  BASE=$$(grep -m1 '^home = ' .venv/pyvenv.cfg | cut -d' ' -f3); \
	  PYTHONPATH="$$PWD/.venv/Lib/site-packages" HF_HUB_OFFLINE=1 "$$BASE/python.exe" -m uvicorn app.main:app --port 8000; \
	else \
	  uv run uvicorn app.main:app --reload --port 8000; \
	fi

# ch03: Milvus Standalone 起停(etcd + minio + milvus)
milvus-up:
	docker compose up -d etcd minio milvus-standalone
	@echo "等待 Milvus 就绪(healthz)..."; \
	for i in $$(seq 1 60); do \
	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; \
	  sleep 3; done; echo "Milvus 未就绪" && exit 1

milvus-down:
	docker compose stop etcd minio milvus-standalone

kb-preview:
	PYTHONPATH=. uv run python scripts/show_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < ../../sql/ch03-seed.sql

# 干净重建知识库:清 MySQL 两表 + drop Milvus collection
kb-reset:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SET FOREIGN_KEY_CHECKS=0; DELETE FROM knowledge_chunks; DELETE FROM qa_extraction_staging; SET FOREIGN_KEY_CHECKS=1;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "KB 已重置。重跑: make kb-build && make kb-vectorize"

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

test:
	uv run pytest -v

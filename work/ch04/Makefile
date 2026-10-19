# Python 强制 UTF-8(中文 Windows 默认 GBK,读带中文的 markdown/jsonl 会 UnicodeDecodeError)
export PYTHONUTF8 = 1

# ch04:模型全在本地 E:\model;所有 target 与 make dev 共用
export HF_HOME ?= E:/model
export HF_HUB_OFFLINE ?= 1

.PHONY: dev milvus-up milvus-down kb-preview kb-build kb-vectorize kb-revectorize kb-mine seed-conv kb-reset eval-retrieval eval-mining eval-query-rewrite eval-rag-prompt judge-check eval-check eval-rag verify-ch04 smoke-ch04 test

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

# ch04: Milvus Standalone 起停(etcd + minio + milvus;按 compose project 作用域)
milvus-up:
	@if docker inspect -f '{{.State.Running}}' mewhelp-milvus 2>/dev/null | grep -qx true; then \
	  echo 'Milvus already running; skip compose up (fixed container name)'; \
	else \
	  docker compose up -d etcd minio milvus-standalone; \
	fi
	@echo 'Waiting for Milvus healthz...'; \
	i=0; \
	until curl -sf http://localhost:9091/healthz >/dev/null 2>&1; do \
	  i=$$((i + 1)); \
	  if [ $$i -ge 60 ]; then echo 'Milvus not ready' >&2; exit 1; fi; \
	  sleep 3; \
	done; \
	echo 'Milvus OK'

milvus-down:
	docker compose stop etcd minio milvus-standalone

kb-preview:
	PYTHONPATH=. uv run python scripts/show_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-revectorize:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "UPDATE knowledge_chunks SET vectorize_status='pending', vector_id=NULL;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "已清空 Milvus 并把全部 chunk 置回 pending。接着跑: make kb-vectorize"

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < ../../sql/ch03-seed.sql

# 干净重建知识库:清 MySQL 两表 + drop Milvus collection
kb-reset:
	docker exec -i mewhelp-mysql mysql -uroot -proot mewhelp -e "SET FOREIGN_KEY_CHECKS=0; DELETE FROM knowledge_chunks; DELETE FROM qa_extraction_staging; SET FOREIGN_KEY_CHECKS=1;"
	PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; m.drop(m.get_client(), 'knowledge')"
	@echo "KB reset. Next: make kb-build && make kb-vectorize"

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

eval-query-rewrite:
	PYTHONPATH=. uv run python scripts/eval_query_rewrite.py

eval-rag-prompt:
	PYTHONPATH=. uv run python scripts/eval_rag_prompt.py

judge-check:
	PYTHONPATH=. uv run python scripts/check_judge.py

eval-check:
	PYTHONPATH=. uv run python scripts/build_eval_set.py

eval-rag:
	PYTHONPATH=. uv run python scripts/eval_ch04.py $(ARGS)

verify-ch04:
	PYTHONPATH=. uv run python scripts/verify_acceptance.py

smoke-ch04:
	PYTHONPATH=. uv run python scripts/smoke_ch04.py

test:
	uv run pytest -v

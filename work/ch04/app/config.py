from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 账号三件套故意不给默认值:缺了启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000

    database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp_ch04_test"

    embed_model: str = "BAAI/bge-m3"      # 本地模型名;首次加载从 HF 下 ~2GB
    embed_device: str = "cpu"             # cpu / cuda:0;非 cpu 才用 fp16 + devices
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = 3              # ch03 /kb 兼容项;ch04 主链路用 rerank_top_k
    retrieval_min_score: float = 0.4      # ch03 兼容项;ch04 用 rerank_min_score

    # ch04 混合检索 + 本地重排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    recall_top_k: int = 50                # dense / BM25 各召回 Top-50
    rerank_top_k: int = 10                # 精排出 Top-10
    rerank_min_score: float = 0.3         # 机械闸:最高 rerank 分低于此 → 证据不足(不落池)
    query_understanding_timeout: float = 5.0  # Query 改写 LLM 超时秒数;超时降级为原句
    selfcheck_timeout: float = 5.0            # 证据自评 LLM 超时秒数;超时 fail-open 放行
    judge_timeout: float = 30.0               # 评估裁判 LLM 超时秒数;超时由评估脚本跳过该题


settings = Settings()

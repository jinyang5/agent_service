from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 账号三件套故意不给默认值:缺了启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000

    database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp_ch03_test"

    embed_model: str = "BAAI/bge-m3"      # 本地模型名;首次加载从 HF 下 ~2GB
    embed_device: str = "cpu"             # cpu / cuda:0;非 cpu 才用 fp16 + devices
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LegacyLens API"
    app_env: str = "production"
    app_debug: bool = False

    openai_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "legacylens-openai-index"
    pinecone_namespace: str = "nshmp-main"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    external_call_retries: int = 3
    external_call_backoff_seconds: float = 0.5
    embedding_cache_size: int = 512
    retrieval_candidate_multiplier: int = 4
    retrieval_max_candidates: int = 40
    retrieval_lexical_weight: float = 0.25
    retrieval_min_hybrid_score: float = 0.35
    rag_max_context_chunks: int = 6
    pinecone_fallback_namespace: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

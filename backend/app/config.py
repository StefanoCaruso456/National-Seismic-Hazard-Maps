from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LegacyLens API"
    app_env: str = "production"
    app_debug: bool = False

    openai_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "legacylens-openai-index"
    pinecone_namespace: str = "nshmp-main:v1"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    enforce_embedding_dimension: bool = True
    external_call_retries: int = 3
    external_call_backoff_seconds: float = 0.5
    embedding_cache_size: int = 512
    retrieval_candidate_multiplier: int = 4
    retrieval_max_candidates: int = 40
    retrieval_lexical_weight: float = 0.25
    retrieval_min_hybrid_score: float = 0.35
    retrieval_focus_term_guardrail_enabled: bool = True
    retrieval_focus_term_absent_cap: float = 0.20
    retrieval_focus_term_partial_coverage_cap: float = 0.45
    retrieval_query_cache_ttl_seconds: float = 20.0
    retrieval_query_cache_max_entries: int = 256
    retrieval_identifier_file_limit: int = 80
    retrieval_identifier_lexical_enabled: bool = True
    retrieval_deterministic_rerank_enabled: bool = True
    retrieval_context_expansion_enabled: bool = True
    retrieval_context_neighbor_lines: int = 6
    retrieval_context_parent_max_lines: int = 220
    retrieval_context_header_lines: int = 18
    hybrid_top_k_default: int = 12
    hybrid_max_candidate_files: int = 50
    gitnexus_enabled: bool = True
    gitnexus_base_url: str = "http://127.0.0.1:4000"
    gitnexus_default_repo: str | None = None
    gitnexus_mcp_command: str = "npx -y gitnexus@latest mcp"
    gitnexus_call_timeout_seconds: float = 30.0
    gitnexus_startup_timeout_seconds: float = 45.0
    gitnexus_bootstrap_enabled: bool = False
    gitnexus_bootstrap_repo_url: str | None = None
    gitnexus_bootstrap_repo_path: str = "/tmp/nshmp-main"
    gitnexus_bootstrap_repo_ref: str | None = None
    gitnexus_analyze_command: str = "gitnexus analyze"
    gitnexus_analyze_timeout_seconds: float = 180.0
    repo_root_override: str | None = None
    rag_max_context_chunks: int = 6
    pinecone_fallback_namespace: str | None = None
    startup_smoke_mode: str = "off"
    startup_smoke_query: str = "startup health probe"
    startup_smoke_top_k: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

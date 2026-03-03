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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

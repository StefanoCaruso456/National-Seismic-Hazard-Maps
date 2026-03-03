from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LegacyLens API"
    app_env: str = "production"
    app_debug: bool = False

    openai_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "legacylens-openai-index"
    pinecone_namespace: str = "nshmp-main"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

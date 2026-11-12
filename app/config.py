"""Application settings loaded from environment variables and the ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ASYNC_DRIVER_PREFIX = "postgresql+asyncpg://"
_PLAIN_PREFIXES = ("postgresql://", "postgres://")


class Settings(BaseSettings):
    """Typed runtime configuration. Every field maps to an UPPER_CASE env var."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Database
    database_url: SecretStr = Field(
        description="URL of the READ-ONLY role, e.g. postgresql://sql_agent_ro:...@host/db",
    )
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_recycle_s: int = Field(default=1800, ge=30)

    # LLM
    llm_provider: str = Field(default="openai", description="init_chat_model provider key")
    llm_model: str = Field(default="gpt-4.1-mini")
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=60.0, gt=0)

    # Agent
    max_attempts: int = Field(default=3, ge=1, le=10)
    row_limit: int = Field(default=50, ge=1, le=10_000)
    statement_timeout_ms: int = Field(default=5000, ge=100, le=600_000)

    @property
    def async_database_url(self) -> str:
        """Return the database URL using the asyncpg driver."""
        return to_async_url(self.database_url.get_secret_value())


def to_async_url(url: str) -> str:
    """Rewrite ``postgresql://`` / ``postgres://`` URLs to ``postgresql+asyncpg://``."""
    if url.startswith(_ASYNC_DRIVER_PREFIX):
        return url
    for prefix in _PLAIN_PREFIXES:
        if url.startswith(prefix):
            return _ASYNC_DRIVER_PREFIX + url[len(prefix) :]
    raise ValueError("DATABASE_URL must be a PostgreSQL URL (postgresql://...)")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()

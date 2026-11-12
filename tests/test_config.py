"""Unit tests for app.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, to_async_url

URL = "postgresql://ro:secret@localhost:5432/shop"


def make_settings(**overrides: object) -> Settings:
    """Build settings without reading the developer's .env file."""
    return Settings(_env_file=None, database_url=URL, **overrides)  # type: ignore[arg-type]


def test_defaults_match_spec() -> None:
    settings = make_settings()
    assert settings.max_attempts == 3
    assert settings.row_limit == 50
    assert settings.statement_timeout_ms == 5000
    assert settings.llm_api_key is None


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("MAX_ATTEMPTS", "5")
    monkeypatch.setenv("ROW_LIMIT", "10")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.max_attempts == 5
    assert settings.row_limit == 10
    assert settings.llm_provider == "anthropic"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-test"


def test_secrets_are_masked_in_repr() -> None:
    settings = make_settings(llm_api_key="sk-very-secret")
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (URL, "postgresql+asyncpg://ro:secret@localhost:5432/shop"),
        ("postgres://ro@h/db", "postgresql+asyncpg://ro@h/db"),
        ("postgresql+asyncpg://ro@h/db", "postgresql+asyncpg://ro@h/db"),
    ],
)
def test_url_is_upgraded_to_asyncpg(raw: str, expected: str) -> None:
    assert to_async_url(raw) == expected


def test_non_postgres_url_is_rejected() -> None:
    with pytest.raises(ValueError):
        to_async_url("mysql://root@localhost/db")


def test_invalid_max_attempts_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(max_attempts=0)

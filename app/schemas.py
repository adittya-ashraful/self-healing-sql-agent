"""Agent state definition, LLM structured output schema, and HTTP API models."""

from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: Required[str]
    db_schema: str
    generated_sql: str
    thought_process: str
    execution_error: str | None
    query_result: list[dict[str, Any]] | None
    iterations: int
    final_answer: str


def make_initial_state(question: str) -> AgentState:
    """Create a fully populated initial :class:`AgentState` for a graph run."""
    return {
        "question": question,
        "db_schema": "",
        "generated_sql": "",
        "thought_process": "",
        "execution_error": None,
        "query_result": None,
        "iterations": 0,
        "final_answer": "",
    }


class SQLGeneration(BaseModel):
    """Structured output schema returned by the SQL generation LLM call."""

    thought_process: str = Field(
        description=(
            "Step-by-step reasoning explaining why this SQL query answers the question "
            "using the provided schema, or how the previous PostgreSQL error was diagnosed and fixed."
        ),
    )
    sql_query: str = Field(
        description=(
            "A single valid, read-only PostgreSQL SELECT (or WITH ... SELECT) statement "
            "using only tables and columns present in the provided schema."
        ),
    )


class QueryRequest(BaseModel):
    """HTTP request body for ``POST /query/stream``."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question to answer against the PostgreSQL database.",
    )


class StreamEvent(BaseModel):
    """Consistent JSON payload emitted on each ``data: <json>`` SSE frame."""

    type: Literal["node_update", "final", "error", "done"]
    node: str | None = None
    attempt: int | None = None
    sql: str | None = None
    error: str | None = None
    thought_process: str | None = None
    answer: str | None = None

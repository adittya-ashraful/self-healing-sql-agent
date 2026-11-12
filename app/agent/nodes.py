"""Async LangGraph node implementations for the Self-Healing SQL Agent."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncEngine

from app.database import execute_safe_query, fetch_db_schema
from app.schemas import AgentState, SQLGeneration

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]

_SQL_SYSTEM_PROMPT = """You are a senior PostgreSQL data engineer.
Rules you MUST follow:
1. Write a single, read-only PostgreSQL SELECT (or WITH ... SELECT) statement.
2. NEVER guess table or column names. Use ONLY the exact names present in the provided schema.
3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, COPY, SET, or multiple statements.
4. For "last month", filter using date_trunc('month', CURRENT_DATE - INTERVAL '1 month') and date_trunc('month', CURRENT_DATE).
5. Exclude cancelled orders (status <> 'cancelled') when computing revenue unless asked otherwise.
"""

_SUMMARY_SYSTEM_PROMPT = """You are a helpful data analyst.
Given a user's question, the PostgreSQL query that was executed, and the exact result rows,
write a clear, concise natural-language answer grounded strictly in the returned data.
Never invent numbers that are not in the result rows."""


def _coerce_sql_generation(raw: Any) -> SQLGeneration:
    """Normalize structured output whether the model returns a ``SQLGeneration`` or a ``dict``."""
    if isinstance(raw, SQLGeneration):
        return raw
    if isinstance(raw, dict):
        return SQLGeneration.model_validate(raw)
    raise TypeError(f"Unexpected structured output type from LLM: {type(raw)!r}")


def _extract_text(message: Any) -> str:
    """Extract plain text from a LangChain AIMessage or string response."""
    if isinstance(message, str):
        return message.strip()
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ]
        return "".join(parts).strip()
    return str(content).strip()


def make_fetch_schema_node(engine: AsyncEngine) -> NodeFn:
    """Create the ``fetch_schema`` node bound to ``engine``."""

    async def fetch_schema_node(state: AgentState) -> dict[str, Any]:
        """Introspect the live PostgreSQL ``public`` schema and store it in state."""
        del state
        schema_text = await fetch_db_schema(engine)
        return {"db_schema": schema_text}

    return fetch_schema_node


def make_generate_sql_node(llm: BaseChatModel) -> NodeFn:
    """Create the ``generate_sql`` node bound to ``llm`` (with structured output)."""
    structured_llm = llm.with_structured_output(SQLGeneration)

    async def generate_sql_node(state: AgentState) -> dict[str, Any]:
        """Generate or self-heal a PostgreSQL query using the live schema and any prior error."""
        question = state["question"]
        db_schema = state.get("db_schema") or "Schema unavailable."
        previous_error = state.get("execution_error")
        previous_sql = state.get("generated_sql") or ""
        iterations = int(state.get("iterations") or 0)

        if previous_error:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Database Schema:\n{db_schema}\n\n"
                f"Previous SQL Query (Attempt {iterations}):\n{previous_sql}\n\n"
                f"PostgreSQL Error:\n{previous_error}\n\n"
                "Analyze the exact PostgreSQL error message against the Database Schema above. "
                "Explain in thought_process what caused the error and how you are fixing it, "
                "then provide the corrected read-only PostgreSQL SELECT query in sql_query."
            )
        else:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Database Schema:\n{db_schema}\n\n"
                "Write a read-only PostgreSQL SELECT query that answers the question using "
                "only the tables and columns in the Database Schema."
            )

        raw_output = await structured_llm.ainvoke(
            [
                SystemMessage(content=_SQL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        parsed = _coerce_sql_generation(raw_output)
        return {
            "generated_sql": parsed.sql_query.strip(),
            "thought_process": parsed.thought_process.strip(),
            "iterations": iterations + 1,
        }

    return generate_sql_node


def make_execute_sql_node(
    engine: AsyncEngine,
    *,
    row_limit: int = 50,
    statement_timeout_ms: int = 5000,
) -> NodeFn:
    """Create the ``execute_sql`` node bound to ``engine`` and safety limits."""

    async def execute_sql_node(state: AgentState) -> dict[str, Any]:
        """Run ``generated_sql`` through the safe read-only executor."""
        sql = state.get("generated_sql") or ""
        result = await execute_safe_query(
            sql,
            engine,
            row_limit=row_limit,
            statement_timeout_ms=statement_timeout_ms,
        )
        if result.status == "error":
            return {
                "execution_error": result.error or "Unknown database execution error.",
                "query_result": None,
            }
        return {
            "query_result": result.data or [],
            "execution_error": None,
        }

    return execute_sql_node


def make_summarize_node(llm: BaseChatModel) -> NodeFn:
    """Create the ``summarize`` node bound to ``llm``."""

    async def summarize_node(state: AgentState) -> dict[str, Any]:
        """Turn the executed SQL and result rows into a natural-language final answer."""
        question = state["question"]
        sql = state.get("generated_sql") or ""
        rows = state.get("query_result") or []
        rows_json = json.dumps(rows, indent=2, default=str)

        prompt = (
            f"Question: {question}\n\n"
            f"Executed SQL:\n{sql}\n\n"
            f"Result Rows ({len(rows)} row(s)):\n{rows_json}\n\n"
            "Provide a clear, direct answer to the user's question based on these rows."
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return {"final_answer": _extract_text(response)}

    return summarize_node


def make_give_up_node() -> NodeFn:
    """Create the ``give_up`` node used when ``iterations >= max_attempts``."""

    async def give_up_node(state: AgentState) -> dict[str, Any]:
        """Return an honest failure explanation including the last SQL and Postgres error."""
        attempts = int(state.get("iterations") or 0)
        last_sql = state.get("generated_sql") or "<none>"
        last_error = state.get("execution_error") or "<unknown error>"
        message = (
            f"I was unable to answer your question after {attempts} attempt(s) because the "
            "generated SQL query kept failing.\n\n"
            f"Last SQL attempted:\n{last_sql}\n\n"
            f"Last PostgreSQL error:\n{last_error}"
        )
        return {"final_answer": message}

    return give_up_node

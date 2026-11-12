"""API tests for GET /health and POST /query/stream SSE streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.graph import build_graph
from app.config import get_settings
from app.database import create_db_engine
from app.main import app, get_graph
from app.schemas import SQLGeneration
from tests.fakes import ScriptedSQLChatModel
from tests.test_graph import BAD_CREATED_AT_SQL, GOOD_REVENUE_SQL


def _parse_sse_lines(raw_text: str) -> list[dict[str, Any]]:
    """Parse SSE frames (``data: <json>\\n\\n``) into dicts."""
    frames: list[dict[str, Any]] = []
    for block in raw_text.split("\n\n"):
        line = block.strip()
        if not line:
            continue
        assert line.startswith("data: "), f"Unexpected SSE frame: {line!r}"
        payload = json.loads(line[len("data: ") :])
        assert set(payload.keys()) == {
            "type",
            "node",
            "attempt",
            "sql",
            "error",
            "thought_process",
            "answer",
        }
        frames.append(payload)
    return frames


@pytest_asyncio.fixture(scope="module")
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_db_engine(get_settings())
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.integration
async def test_health_endpoint(db_engine: AsyncEngine) -> None:
    app.state.db_engine = db_engine
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.integration
async def test_query_stream_sse_self_healing_event_order(db_engine: AsyncEngine) -> None:
    fake_llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Try querying orders.created_at first.",
                sql_query=BAD_CREATED_AT_SQL,
            ),
            SQLGeneration(
                thought_process="Fix created_at -> order_date after reading Postgres error.",
                sql_query=GOOD_REVENUE_SQL,
            ),
        ],
        summary_text="Last month's revenue was $834.70.",
    )
    compiled_graph = build_graph(fake_llm, db_engine, max_attempts=3)
    app.dependency_overrides[get_graph] = lambda: compiled_graph

    try:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            client.stream(
                "POST",
                "/query/stream",
                json={"question": "What was last month's revenue?"},
            ) as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body_text = (await response.aread()).decode("utf-8")
    finally:
        app.dependency_overrides.clear()

    events = _parse_sse_lines(body_text)
    node_sequence = [(e["type"], e["node"]) for e in events]
    assert node_sequence == [
        ("node_update", "fetch_schema"),
        ("node_update", "generate_sql"),
        ("node_update", "execute_sql"),
        ("node_update", "generate_sql"),
        ("node_update", "execute_sql"),
        ("final", "summarize"),
        ("done", None),
    ]

    # Attempt 1 failed with the real Postgres column error
    first_exec = events[2]
    assert first_exec["attempt"] == 1
    assert first_exec["error"] is not None
    assert "created_at" in first_exec["error"]

    # Attempt 2 fixed the query and succeeded
    second_gen = events[3]
    assert second_gen["attempt"] == 2
    assert "order_date" in (second_gen["sql"] or "")

    second_exec = events[4]
    assert second_exec["attempt"] == 2
    assert second_exec["error"] is None

    # Final event carries the natural-language answer
    final_ev = events[5]
    assert final_ev["type"] == "final"
    assert final_ev["answer"] == "Last month's revenue was $834.70."

    # Stream always ends with done
    assert events[-1]["type"] == "done"


async def test_query_stream_emits_error_and_done_on_unexpected_failure() -> None:
    class BrokenGraph:
        async def astream(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "updates", "data": {"fetch_schema": {"db_schema": "Table orders"}}}
            raise RuntimeError("Unexpected LLM transport failure")

    app.dependency_overrides[get_graph] = lambda: BrokenGraph()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/query/stream",
                json={"question": "What was last month's revenue?"},
            )
            assert resp.status_code == 200
            events = _parse_sse_lines(resp.text)
    finally:
        app.dependency_overrides.clear()

    assert [e["type"] for e in events] == ["node_update", "error", "done"]
    assert "Unexpected LLM transport failure" in (events[1]["error"] or "")

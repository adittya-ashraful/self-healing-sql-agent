"""Integration and unit tests for the Self-Healing SQL Agent LangGraph workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.graph import build_graph
from app.config import get_settings
from app.database import create_db_engine
from app.schemas import SQLGeneration, make_initial_state
from tests.fakes import ScriptedSQLChatModel

GOOD_REVENUE_SQL = """
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.order_date >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month')::date
  AND o.order_date < date_trunc('month', CURRENT_DATE)::date
  AND o.status <> 'cancelled'
""".strip()

BAD_CREATED_AT_SQL = """
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.created_at >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
  AND o.created_at < date_trunc('month', CURRENT_DATE)
  AND o.status <> 'cancelled'
""".strip()


@pytest_asyncio.fixture(scope="module")
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_db_engine(get_settings())
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.integration
async def test_graph_succeeds_on_attempt_1_and_summarizes(db_engine: AsyncEngine) -> None:
    fake_llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Query last month's non-cancelled revenue using orders.order_date.",
                sql_query=GOOD_REVENUE_SQL,
            )
        ],
        summary_text="Last month's revenue was $834.70 across non-cancelled orders.",
    )
    graph = build_graph(fake_llm, db_engine, max_attempts=3)

    state = await graph.ainvoke(make_initial_state("What was last month's revenue?"))
    assert state["iterations"] == 1
    assert state["execution_error"] is None
    assert state["query_result"] == [{"revenue": 834.7}]
    assert "834.70" in state["final_answer"]


@pytest.mark.integration
async def test_graph_self_heals_from_created_at_to_order_date_on_attempt_2(
    db_engine: AsyncEngine,
) -> None:
    fake_llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Filter orders by o.created_at for last month.",
                sql_query=BAD_CREATED_AT_SQL,
            ),
            SQLGeneration(
                thought_process=(
                    "Postgres reported column o.created_at does not exist; "
                    "the schema shows orders uses order_date instead."
                ),
                sql_query=GOOD_REVENUE_SQL,
            ),
        ],
        summary_text="Last month's revenue was $834.70.",
    )
    graph = build_graph(fake_llm, db_engine, max_attempts=3)

    state = await graph.ainvoke(make_initial_state("What was last month's revenue?"))
    assert state["iterations"] == 2
    assert state["execution_error"] is None
    assert state["query_result"] == [{"revenue": 834.7}]
    assert state["final_answer"] == "Last month's revenue was $834.70."

    # Verify the second LLM call received the first query and the real Postgres error
    assert len(fake_llm.structured_prompts) == 2
    second_prompt = fake_llm.structured_prompts[1]
    assert "created_at" in second_prompt
    assert "does not exist" in second_prompt


@pytest.mark.integration
async def test_graph_fails_3_times_and_gives_up_honestly(db_engine: AsyncEngine) -> None:
    fake_llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Attempt 1 using non-existent column created_at.",
                sql_query="SELECT created_at FROM orders",
            ),
            SQLGeneration(
                thought_process="Attempt 2 using non-existent column ordered_on.",
                sql_query="SELECT ordered_on FROM orders",
            ),
            SQLGeneration(
                thought_process="Attempt 3 using non-existent column timestamp.",
                sql_query="SELECT timestamp_col FROM orders",
            ),
        ],
    )
    graph = build_graph(fake_llm, db_engine, max_attempts=3)

    state = await graph.ainvoke(make_initial_state("When were orders placed?"))
    assert state["iterations"] == 3
    assert state["query_result"] is None
    assert state["execution_error"] is not None
    assert "timestamp_col" in state["execution_error"]
    assert "unable to answer" in state["final_answer"].lower()
    assert "3 attempt(s)" in state["final_answer"]
    assert "SELECT timestamp_col FROM orders" in state["final_answer"]
    # Summarize must NOT have been called when giving up
    assert len(fake_llm.summary_prompts) == 0

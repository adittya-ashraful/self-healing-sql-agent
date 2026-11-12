"""Unit tests for app.agent.nodes."""

from __future__ import annotations

from app.agent.nodes import (
    make_generate_sql_node,
    make_give_up_node,
    make_summarize_node,
)
from app.schemas import SQLGeneration, make_initial_state
from tests.fakes import ScriptedSQLChatModel


async def test_generate_sql_first_pass_uses_question_and_schema() -> None:
    llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Use orders.order_date",
                sql_query="SELECT count(*) FROM orders",
            )
        ]
    )
    node = make_generate_sql_node(llm)
    state = make_initial_state("How many orders are there?")
    state["db_schema"] = "Table orders:\n  - id (integer)\n  - order_date (date)"

    update = await node(state)
    assert update["generated_sql"] == "SELECT count(*) FROM orders"
    assert update["thought_process"] == "Use orders.order_date"
    assert update["iterations"] == 1
    assert "Table orders:" in llm.structured_prompts[0]
    assert "PostgreSQL Error:" not in llm.structured_prompts[0]


async def test_generate_sql_repair_pass_includes_previous_sql_and_error() -> None:
    llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process="Replace created_at with order_date per schema.",
                sql_query="SELECT order_date FROM orders",
            )
        ]
    )
    node = make_generate_sql_node(llm)
    state = {
        "question": "List order dates",
        "db_schema": "Table orders:\n  - order_date (date)",
        "generated_sql": "SELECT created_at FROM orders",
        "execution_error": 'column "created_at" does not exist',
        "iterations": 1,
    }

    update = await node(state)
    assert update["generated_sql"] == "SELECT order_date FROM orders"
    assert update["iterations"] == 2
    prompt = llm.structured_prompts[0]
    assert "SELECT created_at FROM orders" in prompt
    assert 'column "created_at" does not exist' in prompt


async def test_summarize_node_produces_final_answer() -> None:
    llm = ScriptedSQLChatModel(summary_text="There were 8 orders totaling $834.70.")
    node = make_summarize_node(llm)
    state = make_initial_state("What was last month's revenue?")
    state["generated_sql"] = "SELECT 834.70 AS revenue"
    state["query_result"] = [{"revenue": 834.70}]

    update = await node(state)
    assert update["final_answer"] == "There were 8 orders totaling $834.70."
    assert "834.7" in llm.summary_prompts[0]


async def test_give_up_node_includes_last_sql_and_last_error() -> None:
    node = make_give_up_node()
    state = make_initial_state("Broken question")
    state["iterations"] = 3
    state["generated_sql"] = "SELECT nope FROM orders"
    state["execution_error"] = 'column "nope" does not exist'

    update = await node(state)
    answer = update["final_answer"]
    assert "3 attempt(s)" in answer
    assert "SELECT nope FROM orders" in answer
    assert 'column "nope" does not exist' in answer

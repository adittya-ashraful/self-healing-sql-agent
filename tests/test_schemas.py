"""Unit tests for app.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import QueryRequest, SQLGeneration, StreamEvent, make_initial_state


def test_make_initial_state_has_all_required_keys() -> None:
    state = make_initial_state("What was last month's revenue?")
    assert state["question"] == "What was last month's revenue?"
    assert state["db_schema"] == ""
    assert state["generated_sql"] == ""
    assert state["thought_process"] == ""
    assert state["execution_error"] is None
    assert state["query_result"] is None
    assert state["iterations"] == 0
    assert state["final_answer"] == ""


def test_sql_generation_structured_output_validation() -> None:
    out = SQLGeneration(
        thought_process="Join orders and order_items using order_date.",
        sql_query="SELECT SUM(quantity * unit_price) FROM order_items",
    )
    assert out.thought_process.startswith("Join")
    assert "SELECT" in out.sql_query


def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_stream_event_serialization() -> None:
    ev = StreamEvent(
        type="node_update",
        node="generate_sql",
        attempt=1,
        sql="SELECT 1",
        thought_process="Testing",
    )
    payload = ev.model_dump()
    assert set(payload.keys()) == {
        "type",
        "node",
        "attempt",
        "sql",
        "error",
        "thought_process",
        "answer",
    }
    assert payload["error"] is None
    assert payload["answer"] is None

"""FastAPI application with lifespan resource management and SSE streaming endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from app.agent.graph import build_chat_model, build_graph
from app.config import get_settings
from app.database import create_db_engine
from app.schemas import QueryRequest, StreamEvent, make_initial_state


def _format_sse(event: StreamEvent) -> str:
    """Serialize a :class:`StreamEvent` as a single Server-Sent Event frame."""
    return f"data: {event.model_dump_json()}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build DB engine, LLM, and compiled graph on startup; dispose DB pool on shutdown."""
    settings = get_settings()
    engine = create_db_engine(settings)
    try:
        # Verify database connectivity before accepting traffic.
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")

        llm = build_chat_model(settings)
        graph = build_graph(llm, engine, settings=settings)

        app.state.settings = settings
        app.state.db_engine = engine
        app.state.llm = llm
        app.state.graph = graph
        yield
    finally:
        await engine.dispose()


app = FastAPI(
    title="Self-Healing SQL Agent",
    version="0.1.0",
    description="Streaming Text-to-SQL agent for PostgreSQL with automatic error recovery.",
    lifespan=lifespan,
)


def get_graph(request: Request) -> CompiledStateGraph:
    """FastAPI dependency that provides the compiled LangGraph instance from ``app.state``."""
    return request.app.state.graph


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness and database connectivity check."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is not None:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    return {"status": "ok"}


async def _stream_graph_events(
    graph: CompiledStateGraph,
    question: str,
    request: Request | None = None,
) -> AsyncIterator[str]:
    """Stream per-node LangGraph updates as SSE ``data: <json>`` frames."""
    initial_state = make_initial_state(question)
    current_attempt = 0
    current_sql: str | None = None
    current_thought: str | None = None

    try:
        async for part in graph.astream(
            initial_state,
            stream_mode="updates",
            version="v2",
        ):
            if request is not None and await request.is_disconnected():
                return

            if part.get("type") != "updates":
                continue

            updates_by_node: dict[str, dict[str, Any]] = part.get("data") or {}
            for node_name, delta in updates_by_node.items():
                if not isinstance(delta, dict):
                    continue

                if "iterations" in delta and delta["iterations"] is not None:
                    current_attempt = int(delta["iterations"])
                if delta.get("generated_sql"):
                    current_sql = str(delta["generated_sql"])
                if delta.get("thought_process"):
                    current_thought = str(delta["thought_process"])

                is_terminal = node_name in ("summarize", "give_up")
                event_type = "final" if is_terminal else "node_update"

                event = StreamEvent(
                    type=event_type,
                    node=node_name,
                    attempt=current_attempt if current_attempt > 0 else None,
                    sql=delta.get(
                        "generated_sql", current_sql if node_name != "fetch_schema" else None
                    ),
                    error=delta.get("execution_error"),
                    thought_process=delta.get(
                        "thought_process", current_thought if node_name == "generate_sql" else None
                    ),
                    answer=delta.get("final_answer"),
                )
                yield _format_sse(event)
    except Exception as exc:
        yield _format_sse(
            StreamEvent(
                type="error",
                attempt=current_attempt if current_attempt > 0 else None,
                sql=current_sql,
                error=str(exc),
            )
        )
    finally:
        if request is None or not await request.is_disconnected():
            yield _format_sse(
                StreamEvent(
                    type="done",
                    attempt=current_attempt if current_attempt > 0 else None,
                )
            )


@app.post("/query/stream")
async def query_stream(
    body: QueryRequest,
    request: Request,
    graph: CompiledStateGraph = Depends(get_graph),
) -> StreamingResponse:
    """Execute the Self-Healing SQL Agent and stream every node update via SSE."""
    return StreamingResponse(
        _stream_graph_events(graph, body.question, request=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

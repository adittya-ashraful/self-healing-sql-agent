"""LangGraph StateGraph wiring and provider-agnostic LLM factory."""

from __future__ import annotations

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.nodes import (
    make_execute_sql_node,
    make_fetch_schema_node,
    make_generate_sql_node,
    make_give_up_node,
    make_summarize_node,
)
from app.config import Settings, get_settings
from app.database import create_db_engine
from app.schemas import AgentState

RouteTarget = Literal["summarize", "give_up", "generate_sql"]


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Instantiate a provider-agnostic LangChain chat model from :class:`Settings`."""
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {
        "temperature": cfg.llm_temperature,
        "timeout": cfg.llm_timeout_s,
    }
    if cfg.llm_api_key is not None:
        kwargs["api_key"] = cfg.llm_api_key.get_secret_value()
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url

    model = init_chat_model(
        cfg.llm_model,
        model_provider=cfg.llm_provider,
        **kwargs,
    )
    return model


def make_post_execution_router(
    max_attempts: int = 3,
):
    """Build the conditional-edge router that runs after ``execute_sql``."""

    def route_after_execution(state: AgentState) -> RouteTarget:
        """Route to ``summarize`` on success, ``give_up`` at max attempts, else retry."""
        error = state.get("execution_error")
        if not error:
            return "summarize"
        iterations = int(state.get("iterations") or 0)
        if iterations >= max_attempts:
            return "give_up"
        return "generate_sql"

    return route_after_execution


def build_graph(
    llm: BaseChatModel,
    engine: AsyncEngine | None = None,
    *,
    settings: Settings | None = None,
    max_attempts: int | None = None,
    row_limit: int | None = None,
    statement_timeout_ms: int | None = None,
) -> CompiledStateGraph:
    """Compile and return the Self-Healing SQL Agent ``StateGraph``."""
    cfg = settings if (settings is not None or engine is None) else None
    if cfg is None and (max_attempts is None or row_limit is None or statement_timeout_ms is None):
        cfg = get_settings()

    db_engine = engine if engine is not None else create_db_engine(cfg)
    effective_max_attempts = (
        max_attempts if max_attempts is not None else (cfg.max_attempts if cfg else 3)
    )
    effective_row_limit = row_limit if row_limit is not None else (cfg.row_limit if cfg else 50)
    effective_timeout_ms = (
        statement_timeout_ms
        if statement_timeout_ms is not None
        else (cfg.statement_timeout_ms if cfg else 5000)
    )

    builder = StateGraph(AgentState)
    builder.add_node("fetch_schema", make_fetch_schema_node(db_engine))
    builder.add_node("generate_sql", make_generate_sql_node(llm))
    builder.add_node(
        "execute_sql",
        make_execute_sql_node(
            db_engine,
            row_limit=effective_row_limit,
            statement_timeout_ms=effective_timeout_ms,
        ),
    )
    builder.add_node("summarize", make_summarize_node(llm))
    builder.add_node("give_up", make_give_up_node())

    builder.add_edge(START, "fetch_schema")
    builder.add_edge("fetch_schema", "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_conditional_edges(
        "execute_sql",
        make_post_execution_router(effective_max_attempts),
        {
            "summarize": "summarize",
            "give_up": "give_up",
            "generate_sql": "generate_sql",
        },
    )
    builder.add_edge("summarize", END)
    builder.add_edge("give_up", END)

    return builder.compile()

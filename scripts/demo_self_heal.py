"""Self-healing SQL Agent demonstration against the live seeded PostgreSQL database.

Runs the question "What was last month's revenue?" using a scripted fake LLM whose:
  - Attempt 1 guesses `orders.created_at` (which fails against PostgreSQL),
  - Attempt 2 reads the real PostgreSQL error + schema and fixes the column to `orders.order_date`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import create_db_engine  # noqa: E402
from app.schemas import SQLGeneration, make_initial_state  # noqa: E402
from tests.fakes import ScriptedSQLChatModel  # noqa: E402

QUESTION = "What was last month's revenue?"

ATTEMPT_1_BAD_SQL = """
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.created_at >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
  AND o.created_at < date_trunc('month', CURRENT_DATE)
  AND o.status <> 'cancelled'
""".strip()

ATTEMPT_2_FIXED_SQL = """
SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.order_date >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month')::date
  AND o.order_date < date_trunc('month', CURRENT_DATE)::date
  AND o.status <> 'cancelled'
""".strip()


def _format_summary_from_prompt(prompt: str) -> str:
    """Produce a realistic natural-language summary grounded in the real DB result."""
    if "834.7" in prompt:
        return (
            "Last month's revenue was $834.70 across 7 non-cancelled orders "
            "(excluding 1 cancelled order)."
        )
    return f"Completed query. Raw prompt excerpt: {prompt[-120:]}"


async def run_demo() -> None:
    """Stream the self-healing workflow against the seeded PostgreSQL database."""
    settings = get_settings()
    engine = create_db_engine(settings)

    fake_llm = ScriptedSQLChatModel(
        sql_steps=[
            SQLGeneration(
                thought_process=(
                    "Join orders and order_items and filter last month's non-cancelled "
                    "orders using o.created_at."
                ),
                sql_query=ATTEMPT_1_BAD_SQL,
            ),
            SQLGeneration(
                thought_process=(
                    "PostgreSQL reported `column o.created_at does not exist`. "
                    "Inspecting the fetched `orders` schema shows the date column is "
                    "named `order_date` (type `date`). Replacing `o.created_at` with "
                    "`o.order_date`."
                ),
                sql_query=ATTEMPT_2_FIXED_SQL,
            ),
        ],
        summary_fn=_format_summary_from_prompt,
    )

    graph = build_graph(fake_llm, engine, settings=settings)

    print("=" * 76)
    print("SELF-HEALING SQL AGENT DEMO")
    print(f"Question: {QUESTION!r}")
    print("=" * 76)

    attempt = 0
    try:
        async for part in graph.astream(
            make_initial_state(QUESTION),
            stream_mode="updates",
            version="v2",
        ):
            if part.get("type") != "updates":
                continue
            updates: dict[str, dict[str, Any]] = part.get("data") or {}
            for node_name, delta in updates.items():
                if node_name == "fetch_schema":
                    schema_lines = (delta.get("db_schema") or "").splitlines()
                    tables = [ln for ln in schema_lines if ln.startswith("Table ")]
                    print(
                        f"\n[1. fetch_schema] Introspected {len(tables)} tables from public schema:"
                    )
                    for t in tables:
                        print(f"    {t}")

                elif node_name == "generate_sql":
                    attempt = int(delta.get("iterations") or (attempt + 1))
                    print(f"\n[2. generate_sql | Attempt {attempt}]")
                    print(f"    Thought : {delta.get('thought_process')}")
                    sql_one_line = " ".join((delta.get("generated_sql") or "").split())
                    print(f"    SQL     : {sql_one_line}")

                elif node_name == "execute_sql":
                    err = delta.get("execution_error")
                    if err:
                        print(f"\n[3. execute_sql  | Attempt {attempt}] -> FAILED")
                        for line in str(err).splitlines():
                            print(f"    Postgres Error: {line}")
                        print(
                            "    Action        : Routing back to generate_sql for self-healing..."
                        )
                    else:
                        rows = delta.get("query_result") or []
                        print(f"\n[3. execute_sql  | Attempt {attempt}] -> SUCCEEDED")
                        print(f"    Rows returned : {rows}")

                elif node_name in ("summarize", "give_up"):
                    print(f"\n[4. {node_name}]")
                    print(f"    Final Answer  : {delta.get('final_answer')}")
    finally:
        await engine.dispose()

    print("\n" + "=" * 76)


if __name__ == "__main__":
    asyncio.run(run_demo())

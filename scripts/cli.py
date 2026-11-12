#!/usr/bin/env python3
"""Interactive & streaming Terminal UI (TUI) for the Self-Healing SQL Agent.

Usage examples:
  # Interactive REPL (live LLM configured in .env)
  uv run python scripts/cli.py

  # Run the scripted self-healing demonstration (created_at -> order_date)
  uv run python scripts/cli.py --demo

  # Ask a single question from the command line
  uv run python scripts/cli.py -q "What was last month's revenue?"
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

with contextlib.suppress(ImportError):
    import readline  # noqa: F401

from pygments import highlight  # noqa: E402
from pygments.formatters import Terminal256Formatter  # noqa: E402
from pygments.lexers import PostgresLexer  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine  # noqa: E402

from app.agent.graph import build_chat_model, build_graph  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.database import create_db_engine, fetch_db_schema  # noqa: E402
from app.schemas import SQLGeneration, make_initial_state  # noqa: E402
from scripts.demo_self_heal import (  # noqa: E402
    ATTEMPT_1_BAD_SQL,
    ATTEMPT_2_FIXED_SQL,
    QUESTION as DEMO_QUESTION,
    _format_summary_from_prompt,
)
from tests.fakes import ScriptedSQLChatModel  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI styling & layout helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Style:
    """ANSI escape sequences for 256-color terminal styling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    CYAN = "\033[38;5;81m"
    BLUE = "\033[38;5;75m"
    GREEN = "\033[38;5;78m"
    AMBER = "\033[38;5;214m"
    RED = "\033[38;5;203m"
    MAGENTA = "\033[38;5;177m"
    GRAY = "\033[38;5;245m"
    WHITE = "\033[38;5;255m"

    BG_CYAN = "\033[48;5;24m\033[38;5;255m\033[1m"
    BG_GREEN = "\033[48;5;22m\033[38;5;120m\033[1m"
    BG_AMBER = "\033[48;5;94m\033[38;5;228m\033[1m"
    BG_RED = "\033[48;5;52m\033[38;5;217m\033[1m"
    BG_MAGENTA = "\033[48;5;53m\033[38;5;225m\033[1m"


EXAMPLE_QUESTIONS: list[str] = [
    "What was last month's revenue?",
    "Who are our top 5 customers by total spend (excluding cancelled orders)?",
    "Which product categories generated the most revenue, and how many units were sold?",
    "List all orders from last month with customer name, order_date, status, and order total.",
    "Which products have generated the highest revenue across all non-cancelled orders?",
]


def _visible_len(text: str) -> int:
    """Return printable character width of ``text`` after stripping ANSI escape codes."""
    return len(_ANSI_RE.sub("", text))


def _term_width(max_width: int = 100) -> int:
    """Return a comfortable box width for the current terminal."""
    cols = shutil.get_terminal_size((92, 24)).columns
    return max(68, min(cols - 2, max_width))


def _pad_visible(text: str, width: int) -> str:
    """Right-pad ``text`` with spaces based on its visible (non-ANSI) length."""
    gap = max(0, width - _visible_len(text))
    return text + (" " * gap)


def _wrap_preserving_lines(text: str, width: int) -> list[str]:
    """Wrap plain text paragraphs to ``width`` characters."""
    out: list[str] = []
    for raw_line in text.splitlines() or [""]:
        stripped = raw_line.rstrip()
        if not stripped:
            out.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=max(20, width),
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.extend(wrapped or [""])
    return out


def render_box(
    title: str,
    lines: list[str],
    *,
    border_color: str = Style.CYAN,
    badge: str | None = None,
    width: int | None = None,
) -> str:
    """Render a Unicode rounded-corner panel around ``lines``."""
    box_w = width or _term_width()
    inner_w = box_w - 4

    header_label = f" {title} "
    if badge:
        header_label = f" {badge} {Style.BOLD}{border_color}{title}{Style.RESET} "
    else:
        header_label = f" {Style.BOLD}{border_color}{title}{Style.RESET} "

    top_fill = max(1, (box_w - 2) - _visible_len(header_label) - 1)
    top = f"{border_color}╭─{Style.RESET}{header_label}{border_color}{'─' * top_fill}╮{Style.RESET}"
    bottom = f"{border_color}╰{'─' * (box_w - 2)}╯{Style.RESET}"

    body: list[str] = [top]
    for line in lines:
        if _visible_len(line) <= inner_w:
            padded = _pad_visible(line, inner_w)
            body.append(f"{border_color}│{Style.RESET} {padded} {border_color}│{Style.RESET}")
        else:
            # Wrap plain lines that exceed the inner width.
            for sub in _wrap_preserving_lines(_ANSI_RE.sub("", line), inner_w):
                padded = _pad_visible(sub, inner_w)
                body.append(f"{border_color}│{Style.RESET} {padded} {border_color}│{Style.RESET}")
    body.append(bottom)
    return "\n".join(body)


def highlight_sql(sql: str, *, max_line_width: int | None = None) -> list[str]:
    """Pretty-format and syntax-highlight a PostgreSQL query for 256-color terminals."""
    clause_split = re.sub(
        r"\b(FROM|LEFT JOIN|INNER JOIN|JOIN|WHERE|AND|OR|GROUP BY|ORDER BY|LIMIT|HAVING)\b",
        r"\n\1",
        " ".join(sql.split()),
        flags=re.IGNORECASE,
    ).strip()
    target_w = max_line_width or (_term_width() - 8)
    wrapped_sql_lines: list[str] = []
    for line in clause_split.splitlines():
        wrapped_sql_lines.extend(
            textwrap.wrap(
                line,
                width=max(30, target_w),
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    formatted = "\n".join(wrapped_sql_lines)
    colored = highlight(formatted, PostgresLexer(), Terminal256Formatter(style="monokai")).rstrip()
    return [f"  {line}" for line in colored.splitlines()]


def render_rows_table(rows: list[dict[str, Any]], *, max_rows: int = 15) -> list[str]:
    """Render query result dicts as an aligned Unicode table."""
    if not rows:
        return [f"{Style.GRAY}(0 rows returned){Style.RESET}"]

    columns = list(rows[0].keys())
    shown = rows[:max_rows]

    col_widths: dict[str, int] = {col: len(str(col)) for col in columns}
    for row in shown:
        for col in columns:
            val_str = "NULL" if row.get(col) is None else str(row.get(col))
            col_widths[col] = min(42, max(col_widths[col], len(val_str)))

    def _fmt_cell(val: Any, width: int) -> str:
        raw = "NULL" if val is None else str(val)
        if len(raw) > width:
            raw = raw[: width - 1] + "…"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return raw.rjust(width)
        return raw.ljust(width)

    top = "┌─" + "─┬─".join("─" * col_widths[c] for c in columns) + "─┐"
    sep = "├─" + "─┼─".join("─" * col_widths[c] for c in columns) + "─┤"
    bot = "└─" + "─┴─".join("─" * col_widths[c] for c in columns) + "─┘"
    hdr = (
        "│ "
        + " │ ".join(
            f"{Style.BOLD}{Style.CYAN}{c.ljust(col_widths[c])}{Style.RESET}" for c in columns
        )
        + " │"
    )

    out = [f"{Style.GRAY}{top}{Style.RESET}", hdr, f"{Style.GRAY}{sep}{Style.RESET}"]
    for row in shown:
        cells = " │ ".join(_fmt_cell(row.get(c), col_widths[c]) for c in columns)
        out.append(f"│ {cells} │")
    out.append(f"{Style.GRAY}{bot}{Style.RESET}")

    if len(rows) > max_rows:
        out.append(f"{Style.GRAY}… showing {max_rows} of {len(rows)} rows{Style.RESET}")
    return out


class LiveSpinner:
    """Async braille spinner shown while waiting for a graph node or DB call."""

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, label: str) -> None:
        self.label = label
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def update(self, label: str) -> None:
        """Update the spinner status text."""
        self.label = label

    async def _spin(self) -> None:
        idx = 0
        start = time.perf_counter()
        while self._running:
            frame = self._FRAMES[idx % len(self._FRAMES)]
            elapsed = time.perf_counter() - start
            line = (
                f"\r{Style.CYAN}{frame}{Style.RESET} "
                f"{Style.BOLD}{self.label}{Style.RESET} "
                f"{Style.GRAY}({elapsed:.1f}s){Style.RESET}\033[K"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            idx += 1
            await asyncio.sleep(0.08)

    async def __aenter__(self) -> LiveSpinner:
        if sys.stdout.isatty():
            self._running = True
            self._task = asyncio.create_task(self._spin())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Banner, schema viewer, and streaming execution
# ---------------------------------------------------------------------------


def _describe_db_target(settings: Settings) -> tuple[str, str]:
    parsed = urlparse(settings.database_url.get_secret_value())
    role = (parsed.username or "unknown").split(".", 1)[0]
    host = f"{parsed.hostname or 'localhost'}:{parsed.port or 5432}{parsed.path or ''}"
    return role, host


def print_banner(settings: Settings, db_latency_ms: float | None, *, mode_label: str) -> None:
    """Print the application header panel."""
    role, host = _describe_db_target(settings)
    db_status = (
        f"{Style.GREEN}● Connected{Style.RESET} ({db_latency_ms:.0f} ms)"
        if db_latency_ms is not None
        else f"{Style.RED}● Unreachable{Style.RESET}"
    )
    lines = [
        f"{Style.BOLD}{Style.WHITE}Self-Healing Text-to-SQL Agent for PostgreSQL{Style.RESET}  "
        f"{Style.GRAY}(LangGraph • FastAPI • asyncpg • sqlglot){Style.RESET}",
        "",
        f"  {Style.GRAY}Mode            :{Style.RESET} {mode_label}",
        f"  {Style.GRAY}Database        :{Style.RESET} {db_status}  {Style.DIM}{host}{Style.RESET}",
        f"  {Style.GRAY}DB Role         :{Style.RESET} {Style.GREEN}{role}{Style.RESET} "
        f"{Style.GRAY}(READ ONLY transaction + AST guard){Style.RESET}",
        f"  {Style.GRAY}Guardrails      :{Style.RESET} max_attempts={settings.max_attempts}  "
        f"row_limit={settings.row_limit}  timeout={settings.statement_timeout_ms}ms",
        "",
        f"  {Style.GRAY}Commands        :{Style.RESET} "
        f"{Style.CYAN}/demo{Style.RESET} self-heal demo  •  "
        f"{Style.CYAN}/examples{Style.RESET} sample queries  •  "
        f"{Style.CYAN}/schema{Style.RESET} inspect DB  •  "
        f"{Style.CYAN}/quit{Style.RESET} exit",
    ]
    print(
        render_box(
            "SELF-HEALING SQL AGENT",
            lines,
            border_color=Style.CYAN,
            badge=f"{Style.BG_CYAN} SQL AGENT {Style.RESET}",
        )
    )


async def print_live_schema(engine: AsyncEngine) -> None:
    """Fetch and display the live database schema in a styled panel."""
    async with LiveSpinner("Introspecting PostgreSQL public schema..."):
        schema_text = await fetch_db_schema(engine)

    styled_lines: list[str] = []
    for line in schema_text.splitlines():
        if line.startswith("Schema:"):
            styled_lines.append(f"{Style.BOLD}{Style.CYAN}{line}{Style.RESET}")
        elif line.startswith("Table "):
            styled_lines.append("")
            styled_lines.append(f"{Style.BOLD}{Style.AMBER}▸ {line}{Style.RESET}")
        else:
            styled_lines.append(f"  {line}")

    print(
        render_box(
            "Live PostgreSQL Schema",
            styled_lines,
            border_color=Style.BLUE,
            badge=f"{Style.BG_CYAN} SCHEMA {Style.RESET}",
        )
    )


async def stream_question_in_ui(
    question: str,
    graph: Any,
    *,
    max_attempts: int,
) -> None:
    """Run a question through ``graph.astream`` and render each node update live."""
    print()
    print(
        render_box(
            "User Question",
            [f"{Style.BOLD}{Style.WHITE}{question}{Style.RESET}"],
            border_color=Style.MAGENTA,
            badge=f"{Style.BG_MAGENTA} QUERY {Style.RESET}",
        )
    )

    started_at = time.perf_counter()
    attempt = 0
    healed = False

    spinner = LiveSpinner("Step 1/4 • Inspecting live PostgreSQL schema (fetch_schema)...")
    await spinner.__aenter__()
    try:
        async for part in graph.astream(
            make_initial_state(question),
            stream_mode="updates",
            version="v2",
        ):
            if part.get("type") != "updates":
                continue

            updates: dict[str, dict[str, Any]] = part.get("data") or {}
            for node_name, delta in updates.items():
                await spinner.__aexit__(None, None, None)

                if node_name == "fetch_schema":
                    schema_text = str(delta.get("db_schema") or "")
                    tables = [
                        ln.removeprefix("Table ").rstrip(":")
                        for ln in schema_text.splitlines()
                        if ln.startswith("Table ")
                    ]
                    fk_count = schema_text.count("FK ->")
                    col_count = sum(
                        1 for ln in schema_text.splitlines() if ln.strip().startswith("- ")
                    )
                    lines = [
                        f"{Style.GREEN}✓{Style.RESET} Discovered "
                        f"{Style.BOLD}{len(tables)} tables{Style.RESET} "
                        f"({', '.join(tables)}), {col_count} columns, {fk_count} foreign keys."
                    ]
                    print(
                        render_box(
                            "Schema Introspection",
                            lines,
                            border_color=Style.BLUE,
                            badge=f"{Style.BG_CYAN} 1 • FETCH_SCHEMA {Style.RESET}",
                        )
                    )
                    spinner = LiveSpinner("Step 2/4 • Generating SQL query with LLM...")
                    await spinner.__aenter__()

                elif node_name == "generate_sql":
                    attempt = int(delta.get("iterations") or (attempt + 1))
                    thought = str(delta.get("thought_process") or "")
                    sql = str(delta.get("generated_sql") or "")
                    is_retry = attempt > 1
                    if is_retry:
                        healed = True

                    badge_style = Style.BG_AMBER if is_retry else Style.BG_CYAN
                    border = Style.AMBER if is_retry else Style.CYAN
                    title = (
                        f"Self-Healing SQL Generation (Attempt {attempt}/{max_attempts})"
                        if is_retry
                        else f"Initial SQL Generation (Attempt {attempt}/{max_attempts})"
                    )

                    lines = [f"{Style.BOLD}Thought Process:{Style.RESET}"]
                    for wrapped in _wrap_preserving_lines(thought, _term_width() - 8):
                        lines.append(f"  {Style.ITALIC}{wrapped}{Style.RESET}")
                    lines.append("")
                    lines.append(f"{Style.BOLD}Generated PostgreSQL:{Style.RESET}")
                    lines.extend(highlight_sql(sql))

                    print(
                        render_box(
                            title,
                            lines,
                            border_color=border,
                            badge=f"{badge_style} 2 • GENERATE_SQL #{attempt} {Style.RESET}",
                        )
                    )
                    spinner = LiveSpinner(
                        f"Step 3/4 • Executing SQL safely in READ ONLY transaction (Attempt {attempt})..."
                    )
                    await spinner.__aenter__()

                elif node_name == "execute_sql":
                    err = delta.get("execution_error")
                    if err:
                        will_retry = attempt < max_attempts
                        action_msg = (
                            f"{Style.AMBER}↻ Feeding exact PostgreSQL error back to LLM "
                            f"(retry {attempt + 1}/{max_attempts})...{Style.RESET}"
                            if will_retry
                            else f"{Style.RED}✕ Max attempts ({max_attempts}) reached → routing to give_up.{Style.RESET}"
                        )
                        err_lines = [
                            f"{Style.BOLD}{Style.RED}✗ PostgreSQL Execution Error:{Style.RESET}",
                        ]
                        for ln in str(err).splitlines():
                            err_lines.append(f"  {Style.RED}{ln}{Style.RESET}")
                        err_lines.append("")
                        err_lines.append(action_msg)

                        print(
                            render_box(
                                f"Execution Failed (Attempt {attempt}/{max_attempts})",
                                err_lines,
                                border_color=Style.RED,
                                badge=f"{Style.BG_RED} 3 • EXECUTE_SQL FAILED {Style.RESET}",
                            )
                        )
                        next_label = (
                            f"Self-healing: regenerating SQL (Attempt {attempt + 1}/{max_attempts})..."
                            if will_retry
                            else "Preparing failure report..."
                        )
                        spinner = LiveSpinner(next_label)
                        await spinner.__aenter__()
                    else:
                        rows: list[dict[str, Any]] = delta.get("query_result") or []
                        ok_lines = [
                            f"{Style.GREEN}✓ Executed in read-only transaction • "
                            f"{len(rows)} row(s) returned{Style.RESET}",
                            "",
                            *render_rows_table(rows),
                        ]
                        print(
                            render_box(
                                f"Query Succeeded (Attempt {attempt}/{max_attempts})",
                                ok_lines,
                                border_color=Style.GREEN,
                                badge=f"{Style.BG_GREEN} 3 • EXECUTE_SQL OK {Style.RESET}",
                            )
                        )
                        spinner = LiveSpinner(
                            "Step 4/4 • Summarizing result into natural language..."
                        )
                        await spinner.__aenter__()

                elif node_name in ("summarize", "give_up"):
                    elapsed = time.perf_counter() - started_at
                    answer = str(delta.get("final_answer") or "")
                    is_give_up = node_name == "give_up"
                    border = Style.RED if is_give_up else Style.GREEN
                    badge = (
                        f"{Style.BG_RED} 4 • GIVE_UP {Style.RESET}"
                        if is_give_up
                        else f"{Style.BG_GREEN} 4 • FINAL ANSWER {Style.RESET}"
                    )
                    heal_tag = (
                        f" • {Style.AMBER}Self-healed in {attempt} attempts{Style.RESET}"
                        if healed and not is_give_up
                        else f" • {attempt} attempt(s)"
                    )
                    ans_lines = [
                        *_wrap_preserving_lines(answer, _term_width() - 6),
                        "",
                        f"{Style.GRAY}Completed in {elapsed:.2f}s{heal_tag}{Style.RESET}",
                    ]
                    print(
                        render_box(
                            "Unable to Answer" if is_give_up else "Answer",
                            ans_lines,
                            border_color=border,
                            badge=badge,
                        )
                    )
    finally:
        await spinner.__aexit__(None, None, None)


def build_demo_graph(engine: AsyncEngine, settings: Settings) -> Any:
    """Build a graph wired with the scripted self-healing LLM (`created_at` -> `order_date`)."""
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
    return build_graph(fake_llm, engine, settings=settings)


def print_examples() -> None:
    """Display numbered sample questions that can be run by typing 1..N."""
    lines = [
        f"{Style.GRAY}Type a number (1–{len(EXAMPLE_QUESTIONS)}) at the prompt to run it immediately:{Style.RESET}",
        "",
    ]
    for idx, q in enumerate(EXAMPLE_QUESTIONS, start=1):
        lines.append(f"  {Style.BOLD}{Style.CYAN}[{idx}]{Style.RESET} {q}")
    print(
        render_box(
            "Sample Questions",
            lines,
            border_color=Style.CYAN,
            badge=f"{Style.BG_CYAN} EXAMPLES {Style.RESET}",
        )
    )


async def check_db_latency(engine: AsyncEngine) -> float | None:
    """Ping the database and return latency in milliseconds, or ``None`` on failure."""
    t0 = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        return (time.perf_counter() - t0) * 1000.0
    except Exception:
        return None


async def async_main() -> int:
    """Entrypoint for the terminal UI."""
    parser = argparse.ArgumentParser(
        description="Interactive & streaming Terminal UI for the Self-Healing SQL Agent."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Run a single natural-language question and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted self-healing demonstration (created_at -> order_date) against the live DB.",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = create_db_engine(settings)

    try:
        latency_ms = await check_db_latency(engine)
        has_api_key = bool(
            (settings.llm_api_key and settings.llm_api_key.get_secret_value().strip())
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if args.demo:
            mode_label = (
                f"{Style.AMBER}Self-Healing Demo (Scripted LLM + Live PostgreSQL){Style.RESET}"
            )
            print_banner(settings, latency_ms, mode_label=mode_label)
            demo_graph = build_demo_graph(engine, settings)
            await stream_question_in_ui(
                DEMO_QUESTION,
                demo_graph,
                max_attempts=settings.max_attempts,
            )
            return 0

        mode_label = (
            f"{Style.GREEN}Live LLM{Style.RESET} ({settings.llm_provider}:{settings.llm_model})"
            if has_api_key
            else f"{Style.AMBER}Demo Mode (no LLM_API_KEY set; type /demo){Style.RESET}"
        )
        print_banner(settings, latency_ms, mode_label=mode_label)

        live_graph = (
            build_graph(build_chat_model(settings), engine, settings=settings)
            if has_api_key
            else None
        )

        if args.question:
            target_graph = live_graph or build_demo_graph(engine, settings)
            await stream_question_in_ui(
                args.question,
                target_graph,
                max_attempts=settings.max_attempts,
            )
            return 0

        # Interactive REPL loop
        while True:
            try:
                raw = await asyncio.to_thread(
                    input,
                    f"\n{Style.BOLD}{Style.CYAN}sql-agent ❯{Style.RESET} ",
                )
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Style.GRAY}Goodbye!{Style.RESET}")
                break

            cmd = raw.strip()
            if not cmd:
                continue

            low = cmd.lower()
            if low in ("/quit", "/exit", "quit", "exit", ":q"):
                print(f"{Style.GRAY}Goodbye!{Style.RESET}")
                break
            if low == "/clear":
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                print_banner(settings, latency_ms, mode_label=mode_label)
                continue
            if low in ("/help", "?"):
                print_examples()
                continue
            if low == "/examples":
                print_examples()
                continue
            if low == "/schema":
                await print_live_schema(engine)
                continue
            if low == "/demo":
                demo_graph = build_demo_graph(engine, settings)
                await stream_question_in_ui(
                    DEMO_QUESTION,
                    demo_graph,
                    max_attempts=settings.max_attempts,
                )
                continue

            if cmd.isdigit():
                idx = int(cmd)
                if 1 <= idx <= len(EXAMPLE_QUESTIONS):
                    cmd = EXAMPLE_QUESTIONS[idx - 1]

            if live_graph is None:
                print(
                    f"{Style.AMBER}No LLM API key detected in .env. Running scripted self-heal demo instead:{Style.RESET}"
                )
                demo_graph = build_demo_graph(engine, settings)
                await stream_question_in_ui(
                    DEMO_QUESTION,
                    demo_graph,
                    max_attempts=settings.max_attempts,
                )
                continue

            await stream_question_in_ui(
                cmd,
                live_graph,
                max_attempts=settings.max_attempts,
            )
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))

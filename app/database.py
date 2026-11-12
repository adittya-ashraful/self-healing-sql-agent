"""Safe async database layer: engine factory, sqlglot AST validator, read-only executor, and schema introspection."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

import sqlglot
from pydantic import BaseModel
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.config import Settings, get_settings

# Allowed top-level AST statement types (WITH ... SELECT parses as Select/Union/etc. with a `with_` clause).
_ALLOWED_ROOT_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
)

# Forbidden AST nodes anywhere in the tree (blocks DML CTEs, SELECT INTO, FOR UPDATE, DDL, etc.).
_FORBIDDEN_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.Grant,
    exp.Revoke,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Lock,
    exp.Execute,
    exp.Kill,
    exp.Analyze,
    exp.Comment,
    exp.Refresh,
)

# PostgreSQL functions with side effects, DoS risk, or filesystem/network access.
_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "set_config",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_stat_file",
        "pg_ls_dir",
        "lo_import",
        "lo_export",
        "lo_unlink",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "nextval",
        "setval",
        "currval",
        "query_to_xml",
    }
)


class QueryExecutionResult(BaseModel):
    """Result returned by :func:`execute_safe_query`.

    Supports both attribute access (``res.status``) and mapping access (``res["status"]``).
    """

    status: Literal["success", "error"]
    data: list[dict[str, Any]] | None = None
    error: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access (`result["status"]`) in addition to attributes."""
        return getattr(self, key)


def create_db_engine(settings: Settings | None = None) -> AsyncEngine:
    """Build an async SQLAlchemy engine with pooling, pre-ping, and statement timeout."""
    cfg = settings or get_settings()
    return create_async_engine(
        cfg.async_database_url,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=cfg.db_pool_recycle_s,
        connect_args={
            # Disable prepared statement caches so Supavisor / PgBouncer transaction
            # pooling (port 6543) and direct Postgres (port 5432) both work cleanly.
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "server_settings": {
                "statement_timeout": str(cfg.statement_timeout_ms),
                "default_transaction_read_only": "on",
            },
        },
    )


def validate_sql(sql: str) -> str | None:
    """Validate that ``sql`` is a single read-only ``SELECT`` / ``WITH ... SELECT`` query.

    Returns ``None`` if valid, or a descriptive error message if rejected.
    """
    if not sql or not sql.strip():
        return "ValidationError: SQL query is empty."

    try:
        statements = sqlglot.parse(sql, read="postgres", error_level=sqlglot.ErrorLevel.RAISE)
    except SqlglotError as exc:
        return f"ValidationError: Invalid PostgreSQL syntax: {exc}"

    if len(statements) != 1 or statements[0] is None:
        return (
            f"ValidationError: Expected exactly 1 SQL statement, got {len(statements)}. "
            "Multi-statement queries are not allowed."
        )

    root = statements[0]
    if not isinstance(root, _ALLOWED_ROOT_TYPES):
        return (
            f"ValidationError: Statement type '{type(root).__name__.upper()}' is not allowed. "
            "Only SELECT or WITH ... SELECT queries are permitted."
        )

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            return (
                f"ValidationError: Forbidden SQL clause or operation '{type(node).__name__.upper()}' "
                "detected. Queries must be strictly read-only."
            )
        if isinstance(node, exp.Func):
            func_name = (
                node.name.lower() if isinstance(node, exp.Anonymous) else node.sql_name().lower()
            )
            if func_name in _FORBIDDEN_FUNCTIONS:
                return f"ValidationError: Function '{func_name}()' is not allowed in read-only queries."

    return None


def _normalize_cell(value: Any) -> Any:
    """Convert Postgres driver values into JSON-safe Python primitives where needed."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _extract_pg_error(exc: BaseException) -> str:
    """Unwrap SQLAlchemy / asyncpg exceptions to recover the raw PostgreSQL error message."""
    if isinstance(exc, DBAPIError) and exc.orig is not None:
        orig = exc.orig
        driver_exc = getattr(orig, "orig", None)
        if driver_exc is not None:
            return str(driver_exc).strip()
        return str(orig).strip()
    return str(exc).strip()


async def execute_safe_query(
    sql: str,
    engine: AsyncEngine,
    *,
    row_limit: int = 50,
    statement_timeout_ms: int = 5000,
) -> QueryExecutionResult:
    """Validate and execute a read-only SQL query inside a read-only transaction.

    Never raises on invalid or failing SQL; returns :class:`QueryExecutionResult`
    with the original PostgreSQL error message so the self-healing loop can fix it.
    """
    validation_error = validate_sql(sql)
    if validation_error is not None:
        return QueryExecutionResult(status="error", data=None, error=validation_error)

    clean_sql = sql.strip().rstrip(";").strip()
    timeout_ms = max(100, int(statement_timeout_ms))

    try:
        async with engine.connect() as conn:
            tx = await conn.begin()
            try:
                await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                await conn.exec_driver_sql(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
                cursor = await conn.exec_driver_sql(clean_sql)
                raw_rows = cursor.mappings().fetchmany(row_limit)
                rows = [
                    {str(col): _normalize_cell(val) for col, val in dict(row).items()}
                    for row in raw_rows
                ]
                return QueryExecutionResult(status="success", data=rows, error=None)
            finally:
                if tx.is_active:
                    await tx.rollback()
    except (SQLAlchemyError, Exception) as exc:
        return QueryExecutionResult(
            status="error",
            data=None,
            error=_extract_pg_error(exc),
        )


_COLUMNS_SQL = """
SELECT
    c.table_name,
    c.column_name,
    c.data_type,
    c.is_nullable
FROM information_schema.columns AS c
JOIN information_schema.tables AS t
  ON t.table_schema = c.table_schema
 AND t.table_name = c.table_name
WHERE c.table_schema = 'public'
  AND t.table_type = 'BASE TABLE'
ORDER BY c.table_name, c.ordinal_position
"""

# NOTE: In PostgreSQL, information_schema.constraint_column_usage hides
# constraints from roles that only hold SELECT (non-owners). Querying
# pg_constraint + information_schema.columns lets a strict read-only role
# still see primary and foreign key relationships.
_PRIMARY_KEYS_SQL = """
SELECT
    cls.relname AS table_name,
    att.attname AS column_name
FROM pg_constraint AS con
JOIN pg_class AS cls ON cls.oid = con.conrelid
JOIN pg_namespace AS nsp ON nsp.oid = cls.relnamespace
JOIN unnest(con.conkey) AS k(attnum) ON true
JOIN pg_attribute AS att ON att.attrelid = cls.oid AND att.attnum = k.attnum
WHERE nsp.nspname = 'public'
  AND con.contype = 'p'
ORDER BY cls.relname, att.attnum
"""

_FOREIGN_KEYS_SQL = """
SELECT
    src_cls.relname AS source_table,
    src_att.attname AS source_column,
    tgt_cls.relname AS target_table,
    tgt_att.attname AS target_column
FROM pg_constraint AS con
JOIN pg_class AS src_cls ON src_cls.oid = con.conrelid
JOIN pg_namespace AS nsp ON nsp.oid = src_cls.relnamespace
JOIN pg_class AS tgt_cls ON tgt_cls.oid = con.confrelid
JOIN unnest(con.conkey, con.confkey) AS k(src_attnum, tgt_attnum) ON true
JOIN pg_attribute AS src_att
  ON src_att.attrelid = src_cls.oid AND src_att.attnum = k.src_attnum
JOIN pg_attribute AS tgt_att
  ON tgt_att.attrelid = tgt_cls.oid AND tgt_att.attnum = k.tgt_attnum
WHERE nsp.nspname = 'public'
  AND con.contype = 'f'
ORDER BY src_cls.relname, src_att.attname
"""


async def fetch_db_schema(engine: AsyncEngine) -> str:
    """Inspect ``information_schema`` for ``public`` and format a compact text summary."""
    async with engine.connect() as conn:
        tx = await conn.begin()
        try:
            await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            col_rows = (await conn.exec_driver_sql(_COLUMNS_SQL)).mappings().all()
            pk_rows = (await conn.exec_driver_sql(_PRIMARY_KEYS_SQL)).mappings().all()
            fk_rows = (await conn.exec_driver_sql(_FOREIGN_KEYS_SQL)).mappings().all()
        finally:
            if tx.is_active:
                await tx.rollback()

    pks: set[tuple[str, str]] = {(str(r["table_name"]), str(r["column_name"])) for r in pk_rows}
    fks: dict[tuple[str, str], str] = {
        (str(r["source_table"]), str(r["source_column"])): (
            f"{r['target_table']}.{r['target_column']}"
        )
        for r in fk_rows
    }

    tables: dict[str, list[str]] = {}
    for row in col_rows:
        table = str(row["table_name"])
        col = str(row["column_name"])
        dtype = str(row["data_type"])
        nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
        tags: list[str] = [dtype, nullable]
        if (table, col) in pks:
            tags.append("PK")
        if (table, col) in fks:
            tags.append(f"FK -> {fks[(table, col)]}")
        tables.setdefault(table, []).append(f"  - {col} ({', '.join(tags)})")

    if not tables:
        return "No tables found in schema 'public'."

    lines: list[str] = ["Schema: public"]
    for table_name, col_lines in tables.items():
        lines.append(f"Table {table_name}:")
        lines.extend(col_lines)
    return "\n".join(lines)

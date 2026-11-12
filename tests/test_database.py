"""Unit and integration tests for app.database."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings
from app.database import (
    create_db_engine,
    execute_safe_query,
    fetch_db_schema,
    validate_sql,
)


@pytest_asyncio.fixture(scope="module")
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Provide a pooled async engine connected as the read-only role."""
    engine = create_db_engine(get_settings())
    try:
        yield engine
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Fast AST validation tests (no network required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, full_name FROM customers",
        "SELECT count(*) AS total FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';",
        """
        WITH monthly AS (
            SELECT id, order_date FROM orders WHERE status <> 'cancelled'
        )
        SELECT count(*) FROM monthly
        """,
        "SELECT id FROM customers UNION ALL SELECT customer_id FROM orders",
    ],
)
def test_validate_sql_accepts_safe_selects(sql: str) -> None:
    assert validate_sql(sql) is None


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "INSERT INTO customers (full_name, email, country) VALUES ('x', 'x@y.com', 'US')",
        "UPDATE orders SET status = 'cancelled'",
        "DELETE FROM order_items",
        "DROP TABLE orders",
        "TRUNCATE TABLE orders",
        "ALTER TABLE orders ADD COLUMN foo int",
        "CREATE TABLE hacked (id int)",
        "COPY customers TO '/tmp/leak.csv'",
        "SET statement_timeout = 0",
        "GRANT ALL ON orders TO public",
        "SELECT 1; SELECT 2",
        "SELECT * FROM orders; DROP TABLE orders;",
        "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
        "SELECT * INTO sneaky_table FROM customers",
        "SELECT * FROM orders FOR UPDATE",
        "SELECT pg_sleep(30)",
        "SELECT set_config('statement_timeout', '0', false)",
    ],
)
def test_validate_sql_rejects_unsafe_or_multi_statement_queries(sql: str) -> None:
    err = validate_sql(sql)
    assert err is not None
    assert "ValidationError" in err


# ---------------------------------------------------------------------------
# Live PostgreSQL integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_execute_safe_query_select_works(db_engine: AsyncEngine) -> None:
    res = await execute_safe_query(
        "SELECT id, full_name, country FROM customers ORDER BY id",
        db_engine,
    )
    assert res.status == "success"
    assert res.error is None
    assert res.data is not None
    assert len(res.data) == 8
    assert res.data[0]["full_name"] == "Alice Johnson"


@pytest.mark.integration
async def test_execute_safe_query_enforces_row_limit(db_engine: AsyncEngine) -> None:
    res = await execute_safe_query(
        "SELECT id FROM order_items ORDER BY id",
        db_engine,
        row_limit=3,
    )
    assert res.status == "success"
    assert res.data is not None
    assert len(res.data) == 3


@pytest.mark.integration
@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DELETE FROM orders WHERE id = 1",
        "UPDATE customers SET country = 'Nowhere'",
        "DROP TABLE order_items",
        "SELECT 1; DELETE FROM orders",
        "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
    ],
)
async def test_execute_safe_query_rejects_writes_and_multistatements(
    db_engine: AsyncEngine,
    unsafe_sql: str,
) -> None:
    res = await execute_safe_query(unsafe_sql, db_engine)
    assert res.status == "error"
    assert res.data is None
    assert res.error is not None
    assert "ValidationError" in res.error


@pytest.mark.integration
async def test_execute_safe_query_returns_real_postgres_error_on_bad_column(
    db_engine: AsyncEngine,
) -> None:
    res = await execute_safe_query(
        "SELECT SUM(oi.quantity * oi.unit_price) AS revenue "
        "FROM orders o JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'",
        db_engine,
    )
    assert res.status == "error"
    assert res.data is None
    assert res.error is not None
    assert "created_at" in res.error
    assert "does not exist" in res.error


@pytest.mark.integration
async def test_fetch_db_schema_includes_tables_columns_and_fks(
    db_engine: AsyncEngine,
) -> None:
    schema_text = await fetch_db_schema(db_engine)
    assert "Table customers:" in schema_text
    assert "Table products:" in schema_text
    assert "Table orders:" in schema_text
    assert "Table order_items:" in schema_text
    assert "order_date" in schema_text
    assert "FK -> customers.id" in schema_text
    assert "FK -> orders.id" in schema_text
    assert "FK -> products.id" in schema_text

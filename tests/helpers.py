"""Small helpers shared by the test suites.

DuckDB's `fetchone()` is typed as `tuple[Any, ...] | None`, so every scalar read in a test needs a
guaranteed-row helper. One helper here beats twenty `assert row is not None` blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb


def row(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> tuple[Any, ...]:
    """Return the single row of a single-row query, or fail the test."""
    result = connection.execute(sql, params or []).fetchone()
    assert result is not None, f"query returned no rows: {sql}"
    return result


def scalar(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> float:
    """Return the first column of a single-row query as a float."""
    return float(row(connection, sql, params)[0])

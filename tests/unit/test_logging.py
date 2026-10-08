"""Structured logging: one JSON object per line, with a stable field set."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from rankstack.logging import JsonFormatter, configure_logging, get_logger, request_id


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="rankstack.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened",
        args=(),
        exc_info=None,
    )


def test_record_becomes_a_single_json_object() -> None:
    record = _record()
    record.extra_field = 42
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "info"
    assert payload["msg"] == "something happened"
    assert payload["logger"] == "rankstack.test"
    assert payload["extra_field"] == 42
    assert payload["ts"].endswith("Z")


def test_request_id_is_bound_and_restored() -> None:
    get_logger("rankstack.test")
    formatter = JsonFormatter()
    with request_id("req-42"):
        inside = json.loads(formatter.format(_record()))
    outside = json.loads(formatter.format(_record()))
    assert inside["request_id"] == "req-42"
    assert outside["request_id"] == "-"


def test_unserialisable_values_do_not_break_logging() -> None:
    record = _record()
    record.weird = object()
    payload = json.loads(JsonFormatter().format(record))
    assert isinstance(payload["weird"], str)


def test_exceptions_are_embedded_not_raised() -> None:
    def _raise() -> None:
        msg = "boom"
        raise ValueError(msg)

    record = _record()
    try:
        _raise()
    except ValueError:
        record.exc_info = sys.exc_info()
    else:
        pytest.fail("expected ValueError")
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_configure_logging_replaces_handlers_and_sets_level() -> None:
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler())
    configure_logging(logging.DEBUG)
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    configure_logging(logging.INFO)

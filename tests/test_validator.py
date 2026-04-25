"""Tests for Validator."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.validator import Validator


@pytest.fixture
def val():
    return Validator()


def test_parse_check_valid(val):
    sources = {"mymod": "def foo(x: int) -> int:\n    return x + 1\n"}
    errors = val.parse_check(sources)
    assert errors == []


def test_parse_check_invalid_syntax(val):
    sources = {"badmod": "def foo(\n  @@invalid\n"}
    errors = val.parse_check(sources)
    assert len(errors) > 0
    assert "badmod" in errors[0]


def test_import_resolution_valid(val):
    sources = {
        "utils": "def helper() -> int:\n    return 42\n",
        "main": "from utils import helper\ndef run() -> int:\n    return helper()\n",
    }
    errors = val.import_resolution_check(sources)
    assert errors == []


def test_import_resolution_missing_function(val):
    # utils.helper is defined, but main imports utils.nonexistent which is not
    sources = {
        "utils": "def helper() -> int:\n    return 42\n",
        "main": "from utils import nonexistent\ndef run() -> int:\n    return nonexistent()\n",
    }
    errors = val.import_resolution_check(sources)
    assert len(errors) > 0
    assert "nonexistent" in errors[0]


def test_full_validate_parse_errors_empty_on_valid(val):
    sources = {"ops": "def add(a: int, b: int) -> int:\n    return a + b\n"}
    parse_errors = val.parse_check(sources)
    assert parse_errors == []


def test_full_validate_syntax_error(val):
    sources = {"bad": "def oops(\n  @@@\n"}
    result = val.full_validate(sources)
    assert result.success is False
    assert len(result.parse_errors) > 0

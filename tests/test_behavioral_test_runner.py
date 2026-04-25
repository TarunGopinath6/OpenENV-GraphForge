"""Tests for BehavioralTestRunner."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.behavioral_test_runner import BehavioralTestRunner
from models import BehavioralTest


@pytest.fixture
def runner():
    return BehavioralTestRunner()


SIMPLE_MODULE = {
    "mymod": "def double(x: int) -> int:\n    return x * 2\n"
}

PASSING_TEST = BehavioralTest(
    test_id="test_double_returns_int",
    description="double returns int",
    hypothesis_source="""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mymod import double

def test_double_returns_int():
    assert isinstance(double(3), int)
""",
)

FAILING_TEST = BehavioralTest(
    test_id="test_double_always_42",
    description="double always returns 42 (will fail)",
    hypothesis_source="""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mymod import double

def test_double_always_42():
    assert double(1) == 42
""",
)


def test_passing_test(runner):
    results = runner.run_tests(SIMPLE_MODULE, [PASSING_TEST])
    assert len(results) == 1
    assert results[0].passed is True


def test_failing_test(runner):
    results = runner.run_tests(SIMPLE_MODULE, [FAILING_TEST])
    assert len(results) == 1
    assert results[0].passed is False


def test_no_tests_returns_empty(runner):
    results = runner.run_tests(SIMPLE_MODULE, [])
    assert results == []


def test_missing_module_import_fails(runner):
    bad_test = BehavioralTest(
        test_id="test_missing",
        description="imports nonexistent module",
        hypothesis_source="""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nonexistent_module import something

def test_it():
    assert something() is not None
""",
    )
    results = runner.run_tests(SIMPLE_MODULE, [bad_test])
    assert len(results) == 1
    assert results[0].passed is False

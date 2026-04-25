"""Tests for TaskBank."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import pytest
from tasks.task_bank import TaskBank


@pytest.fixture
def bank():
    return TaskBank(seed=42)


def test_total_task_count(bank):
    assert len(bank) >= 48


def test_tier_counts(bank):
    assert len(bank.list_tasks(1)) == 16
    assert len(bank.list_tasks(2)) == 24
    assert len(bank.list_tasks(3)) == 12


def test_sample_tier1(bank):
    task = bank.sample(tier=1, seed=0)
    assert task.tier == 1
    assert 10 <= len(task.constraints) <= 20
    assert task.token_budget == 4000
    assert task.turn_cap == 20


def test_sample_tier2(bank):
    task = bank.sample(tier=2, seed=0)
    assert task.tier == 2
    assert len(task.constraints) >= 15
    assert task.token_budget == 8000
    assert task.turn_cap == 35


def test_sample_tier3(bank):
    task = bank.sample(tier=3, seed=0)
    assert task.tier == 3
    assert len(task.constraints) >= 20
    assert task.token_budget == 14000
    assert task.turn_cap == 55


def test_behavioral_tests_are_valid_python(bank):
    for task_id in bank.list_tasks():
        task = bank.get_task(task_id)
        for bt in task.behavioral_tests:
            try:
                ast.parse(bt.hypothesis_source)
            except SyntaxError as e:
                pytest.fail(f"Task {task_id} test {bt.test_id} has invalid Python: {e}")


def test_hidden_constraint_ratio(bank):
    for tier in [1, 2, 3]:
        task = bank.sample(tier=tier, seed=0)
        hidden = sum(1 for c in task.constraints if c.hidden)
        total = len(task.constraints)
        ratio = hidden / total if total > 0 else 0
        expected_min = 0.25 if tier == 1 else 0.30
        assert ratio >= expected_min, f"Tier {tier} hidden ratio {ratio:.2f} < {expected_min}"


def test_get_task_by_id(bank):
    ids = bank.list_tasks(1)
    task = bank.get_task(ids[0])
    assert task.task_id == ids[0]


def test_get_unknown_task_raises(bank):
    with pytest.raises(KeyError):
        bank.get_task("nonexistent_task_id")

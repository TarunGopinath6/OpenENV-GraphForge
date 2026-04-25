"""Tests for RewardEngine."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.reward_engine import RewardEngine
from models import ActionResult, ConstraintSpec, GraphState, MaterializeResult


@pytest.fixture
def re():
    return RewardEngine()


def _cspec(kind, target, value=None, hidden=False):
    return ConstraintSpec(constraint_id=f"{kind}_{target}", kind=kind, target=target, value=value, hidden=hidden)


def test_per_turn_failed_action(re):
    result = ActionResult(success=False, error_kind="duplicate", penalty=0.0)
    reward = re.per_turn_reward(result, tokens_this_turn=100, is_repeat=False)
    assert reward < 0


def test_per_turn_repeat_action(re):
    result = ActionResult(success=True, penalty=0.0)
    reward = re.per_turn_reward(result, tokens_this_turn=50, is_repeat=True)
    assert reward < 0


def test_per_turn_success_not_repeat(re):
    result = ActionResult(success=True, penalty=0.0)
    reward = re.per_turn_reward(result, tokens_this_turn=0, is_repeat=False)
    # base per-turn cost only
    assert reward == pytest.approx(-0.1, abs=0.01)


def test_terminal_reward_no_constraints(re):
    mat = MaterializeResult(success=True, module_sources={})
    reward = re.terminal_reward(
        graph=GraphState(),
        constraints=[],
        test_results=[],
        materialize_result=mat,
        mypy_ok=True,
        tokens_used=100,
        token_budget=4000,
    )
    assert isinstance(reward, float)


def test_terminal_reward_materialization_fail_penalty(re):
    mat = MaterializeResult(success=False, module_sources={}, mypy_errors=["type error"])
    reward = re.terminal_reward(
        graph=GraphState(),
        constraints=[],
        test_results=[],
        materialize_result=mat,
        mypy_ok=False,
        tokens_used=100,
        token_budget=4000,
    )
    assert reward <= -8.0


def test_malformed_action_penalty(re):
    # MALFORMED (-2.0) + PER_TURN (-0.1)
    penalty = re.malformed_action_penalty()
    assert penalty == pytest.approx(-2.1, abs=0.01)

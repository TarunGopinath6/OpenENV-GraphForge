"""Tests for GraphForgeClient."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import GraphAction, GraphForgeState, GraphObservation, GraphState, ConstraintSummary


def test_step_payload_serializes_action():
    from client import GraphForgeClient
    client = GraphForgeClient.__new__(GraphForgeClient)
    action = GraphAction(action_type="add_module", parameters={"name": "core", "responsibility": "orchestration"})
    payload = client._step_payload(action)
    assert payload["action_type"] == "add_module"
    assert payload["parameters"]["name"] == "core"


def test_parse_state_from_dict():
    from client import GraphForgeClient
    client = GraphForgeClient.__new__(GraphForgeClient)
    data = {
        "episode_id": "ep1",
        "step_count": 0,
        "graph": {"modules": [], "nodes": [], "edges": []},
        "task_spec": None,
        "turn_cap": 20,
        "token_budget": 4000,
        "tokens_used": 0,
        "cumulative_reward": 0.0,
        "is_terminal": False,
        "materialization_cache": None,
        "last_mypy_output": None,
        "last_test_results": None,
        "action_history": [],
    }
    state = client._parse_state(data)
    assert isinstance(state, GraphForgeState)
    assert state.turn_cap == 20


def test_parse_result_from_dict():
    from client import GraphForgeClient
    client = GraphForgeClient.__new__(GraphForgeClient)
    obs_data = {
        "done": False,
        "reward": 0.0,
        "metadata": {},
        "episode_id": "ep1",
        "timestep": 1,
        "turn_budget_remaining": 19,
        "token_budget_remaining": 3900,
        "graph_state": {"modules": [], "nodes": [], "edges": []},
        "last_action_result": None,
        "constraint_summary": {"total": 0, "visible": 0, "hidden": 0, "satisfied_visible": 0},
        "visible_constraints": [],
        "available_actions": ["add_module"],
        "task_description": "",
    }
    result = client._parse_result(obs_data)
    assert result.observation.timestep == 1
    assert result.done is False

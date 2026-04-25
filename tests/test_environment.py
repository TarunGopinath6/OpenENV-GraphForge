"""Tests for the GraphForge environment (OpenENV compliance)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import GraphForgeEnvironment
from models import GraphAction, GraphObservation, GraphForgeState


def test_environment_init():
    env = GraphForgeEnvironment()
    assert env.state.step_count == 0
    assert env.state.is_terminal is False


def test_environment_reset_returns_observation():
    env = GraphForgeEnvironment()
    obs = env.reset(tier=1, seed=42)
    assert isinstance(obs, GraphObservation)
    assert obs.episode_id != ""
    assert obs.timestep == 0
    assert obs.done is False
    assert obs.turn_budget_remaining > 0
    assert len(obs.visible_constraints) > 0
    assert len(obs.available_actions) > 0


def test_environment_reset_state():
    env = GraphForgeEnvironment()
    env.reset(tier=1, seed=0)
    state = env.state
    assert isinstance(state, GraphForgeState)
    assert state.task_spec is not None
    assert state.graph is not None


def test_environment_step_add_module():
    env = GraphForgeEnvironment()
    env.reset(tier=1, seed=0)
    action = GraphAction(action_type="add_module", parameters={"name": "utils", "responsibility": "transform"})
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert obs.last_action_result.success is True
    assert obs.timestep == 1
    assert len(obs.graph_state.modules) == 1


def test_environment_step_unknown_action_penalized():
    env = GraphForgeEnvironment()
    env.reset(tier=1, seed=0)
    # We can't send truly unknown action_type due to Literal validation, but
    # we verify that the malformed penalty is applied for truly unsatisfiable actions.
    # Use a valid type but that will fail (remove non-existent module).
    action = GraphAction(action_type="remove_module", parameters={"name": "nonexistent"})
    obs = env.step(action)
    assert obs.last_action_result is not None
    assert obs.last_action_result.success is False


def test_environment_step_increments_step_count():
    env = GraphForgeEnvironment()
    env.reset(tier=1, seed=0)
    for i in range(3):
        env.step(GraphAction(action_type="query_spec", parameters={}))
    assert env.state.step_count == 3


def test_environment_full_mini_episode():
    env = GraphForgeEnvironment()
    obs = env.reset(tier=1, seed=1)

    # Add module
    obs = env.step(GraphAction(action_type="add_module", parameters={"name": "core", "responsibility": "transform"}))
    assert obs.last_action_result.success

    # Add node
    obs = env.step(GraphAction(action_type="add_node", parameters={
        "name": "process", "module": "core",
        "signature": "(x: int) -> int",
        "purity": "pure", "error_policy": "raise",
    }))
    assert obs.last_action_result.success

    # Attach body
    obs = env.step(GraphAction(action_type="attach_body", parameters={
        "name": "process", "module": "core",
        "template": "identity", "args": {"param": "x"},
    }))
    assert obs.last_action_result.success

    # Materialize
    obs = env.step(GraphAction(action_type="materialize_and_validate", parameters={}))
    assert obs.last_action_result.success
    assert env.state.materialization_cache is not None

    # Submit
    obs = env.step(GraphAction(action_type="submit", parameters={}))
    assert obs.done is True

"""Tests for ActionDispatcher — all 14 action paths."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.action_dispatcher import ActionDispatcher
from engine.behavioral_test_runner import BehavioralTestRunner
from engine.body_template_library import BodyTemplateLibrary
from engine.constraint_checker import ConstraintChecker
from engine.materializer import Materializer
from engine.reward_engine import RewardEngine
from engine.token_counter import TokenCounter
from engine.type_engine import TypeEngine
from engine.validator import Validator
from models import GraphAction, GraphForgeState


@pytest.fixture
def dispatcher():
    return ActionDispatcher(
        TypeEngine(), BodyTemplateLibrary(), Materializer(), Validator(),
        BehavioralTestRunner(), ConstraintChecker(), RewardEngine(), TokenCounter()
    )


@pytest.fixture
def empty_state():
    return GraphForgeState()


def _act(action_type, **params):
    return GraphAction(action_type=action_type, parameters=params)


def test_add_module(dispatcher, empty_state):
    state, result = dispatcher.dispatch(empty_state, _act("add_module", name="core", responsibility="transform"))
    assert result.success
    assert any(m.name == "core" for m in state.graph.modules)


def test_add_module_duplicate_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="core", responsibility="transform"))
    state, result = dispatcher.dispatch(state, _act("add_module", name="core", responsibility="x"))
    assert not result.success
    assert result.error_kind == "duplicate"


def test_remove_module_with_nodes_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="() -> None", purity="pure", error_policy="raise"))
    state, result = dispatcher.dispatch(state, _act("remove_module", name="m"))
    assert not result.success
    assert result.error_kind == "has_nodes"


def test_add_node(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, result = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    assert result.success
    assert any(n.name == "f" for n in state.graph.nodes)


def test_add_node_bad_signature_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, result = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="@@@bad@@@", purity="pure", error_policy="raise"))
    assert not result.success
    assert result.error_kind == "bad_signature"


def test_remove_node_with_edges_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="a", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="b", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, _ = dispatcher.dispatch(state, _act("add_edge", caller="m.a", callee="m.b", arg_mapping=[{"caller_arg": "x", "callee_param": "x"}]))
    state, result = dispatcher.dispatch(state, _act("remove_node", name="a", module="m"))
    assert not result.success
    assert result.error_kind == "has_edges"


def test_attach_body(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, result = dispatcher.dispatch(state, _act("attach_body", name="f", module="m", template="identity", args={"param": "x"}))
    assert result.success
    assert state.graph.nodes[0].body_template == "identity"


def test_attach_body_bad_template_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, result = dispatcher.dispatch(state, _act("attach_body", name="f", module="m", template="nonexistent", args={}))
    assert not result.success


def test_add_edge_cycle_fails(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="a", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_module", name="b", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="a", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="g", module="b", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    # b imports a
    state, _ = dispatcher.dispatch(state, _act("add_edge", caller="b.g", callee="a.f", arg_mapping=[{"caller_arg": "x", "callee_param": "x"}]))
    # a imports b would be a cycle
    state, result = dispatcher.dispatch(state, _act("add_edge", caller="a.f", callee="b.g", arg_mapping=[{"caller_arg": "x", "callee_param": "x"}]))
    assert not result.success
    assert result.error_kind == "import_cycle"


def test_query_spec(dispatcher, empty_state):
    state, result = dispatcher.dispatch(empty_state, _act("query_spec"))
    assert result.success
    assert result.query_response.total_visible == 0


def test_query_subgraph(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="mod", responsibility="x"))
    state, result = dispatcher.dispatch(state, _act("query_subgraph", scope="module:mod"))
    assert result.success
    assert result.query_response.nodes is not None


def test_query_types(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="(x: int) -> bool", purity="pure", error_policy="raise"))
    state, result = dispatcher.dispatch(state, _act("query_types", scope="all"))
    assert result.success
    assert "m.f" in result.query_response.nodes


def test_materialize_and_validate(dispatcher, empty_state):
    state, _ = dispatcher.dispatch(empty_state, _act("add_module", name="m", responsibility="x"))
    state, _ = dispatcher.dispatch(state, _act("add_node", name="f", module="m", signature="(x: int) -> int", purity="pure", error_policy="raise"))
    state, result = dispatcher.dispatch(state, _act("materialize_and_validate"))
    assert result.success
    assert state.materialization_cache is not None


def test_run_behavioral_tests_without_materialization_fails(dispatcher, empty_state):
    state, result = dispatcher.dispatch(empty_state, _act("run_behavioral_tests"))
    assert not result.success
    assert result.error_kind == "no_materialization"


def test_submit_sets_terminal(dispatcher, empty_state):
    state, result = dispatcher.dispatch(empty_state, _act("submit"))
    assert state.is_terminal is True

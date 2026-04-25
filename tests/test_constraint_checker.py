"""Tests for ConstraintChecker."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.constraint_checker import ConstraintChecker
from models import ConstraintSpec, GraphEdge, GraphModule, GraphNode, GraphState, NodeSignature, ParamSpec


@pytest.fixture
def cc():
    return ConstraintChecker()


def _cspec(kind, target, value=None):
    return ConstraintSpec(constraint_id=f"{kind}_{target}", kind=kind, target=target, value=value)


def _graph_with_node():
    return GraphState(
        modules=[GraphModule(name="utils", responsibility="transform")],
        nodes=[GraphNode(name="parse", module="utils", signature=NodeSignature(
            params=[ParamSpec(name="x", type_annotation="str")], return_type="int"
        ))],
        edges=[],
    )


def test_node_exists_true(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("node_exists", "utils.parse")) is True


def test_node_exists_false(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("node_exists", "utils.missing")) is False


def test_node_absent_true(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("node_absent", "utils.nonexistent")) is True


def test_edge_exists_false_when_no_edges(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("edge_exists", "utils.parse::utils.parse")) is False


def test_module_count(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("module_count", "", value=1)) is True
    assert cc.check_one(graph, _cspec("module_count", "", value=2)) is False


def test_return_type_match(cc):
    graph = _graph_with_node()
    assert cc.check_one(graph, _cspec("return_type", "utils.parse", value="int")) is True
    assert cc.check_one(graph, _cspec("return_type", "utils.parse", value="str")) is False


def test_count_satisfied(cc):
    graph = _graph_with_node()
    constraints = [
        _cspec("node_exists", "utils.parse"),
        _cspec("node_exists", "utils.missing"),
        _cspec("module_count", "", value=1),
    ]
    satisfied, total = cc.count_satisfied(graph, constraints)
    assert total == 3
    assert satisfied == 2


def test_acyclic_imports_empty_graph(cc):
    graph = GraphState()
    assert cc.check_one(graph, _cspec("acyclic_imports", "")) is True

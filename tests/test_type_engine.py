"""Tests for TypeEngine."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.type_engine import TypeEngine, SignatureParseError
from models import EdgeArgMapping, GraphEdge, GraphModule, GraphNode, GraphState, NodeSignature, ParamSpec


@pytest.fixture
def te():
    return TypeEngine()


def test_parse_valid_signature(te):
    sig = te.parse_signature("(x: int, y: str) -> bool")
    assert sig.return_type == "bool"
    assert len(sig.params) == 2
    assert sig.params[0].name == "x"
    assert sig.params[0].type_annotation == "int"
    assert sig.params[1].name == "y"
    assert sig.params[1].type_annotation == "str"


def test_parse_no_args_signature(te):
    sig = te.parse_signature("() -> None")
    assert sig.return_type == "None"
    assert len(sig.params) == 0


def test_parse_invalid_signature_raises(te):
    with pytest.raises(SignatureParseError):
        te.parse_signature("(@@ bad @@)")


def test_detect_any_contamination_true(te):
    node = GraphNode(
        name="f", module="m",
        signature=NodeSignature(params=[ParamSpec(name="x", type_annotation="Any")], return_type="int")
    )
    assert te.detect_any_contamination(node) is True


def test_detect_any_contamination_false(te):
    node = GraphNode(
        name="f", module="m",
        signature=NodeSignature(params=[ParamSpec(name="x", type_annotation="int")], return_type="bool")
    )
    assert te.detect_any_contamination(node) is False


def test_import_cycle_detection(te):
    graph = GraphState(
        modules=[GraphModule(name="a", responsibility="x"), GraphModule(name="b", responsibility="y")],
        nodes=[],
        edges=[GraphEdge(caller="b.g", callee="a.f")],
    )
    # Adding a -> b would create a cycle (b already imports a)
    proposed = GraphEdge(caller="a.f", callee="b.g")
    assert te.has_import_cycle(graph, proposed) is True


def test_no_import_cycle_same_module(te):
    graph = GraphState(
        modules=[GraphModule(name="a", responsibility="x")],
        nodes=[],
        edges=[],
    )
    proposed = GraphEdge(caller="a.f", callee="a.g")
    assert te.has_import_cycle(graph, proposed) is False

"""Tests for Materializer and RoundTripParser."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import pytest
from engine.materializer import Materializer
from engine.round_trip_parser import RoundTripParser
from models import GraphEdge, GraphModule, GraphNode, GraphState, NodeSignature, ParamSpec


@pytest.fixture
def graph():
    return GraphState(
        modules=[GraphModule(name="ops", responsibility="math operations")],
        nodes=[
            GraphNode(
                name="add",
                module="ops",
                decl_order=0,
                signature=NodeSignature(
                    params=[ParamSpec(name="a", type_annotation="int"), ParamSpec(name="b", type_annotation="int")],
                    return_type="int",
                ),
                body_template="add_two",
                body_template_args={"a": "a", "b": "b"},
            ),
            GraphNode(
                name="double",
                module="ops",
                decl_order=1,
                signature=NodeSignature(
                    params=[ParamSpec(name="x", type_annotation="int")],
                    return_type="int",
                ),
                body_template="multiply",
                body_template_args={"a": "x", "b": "2"},
            ),
        ],
        edges=[],
    )


def test_materialize_success(graph):
    mat = Materializer()
    result = mat.materialize(graph)
    assert result.success is True
    assert "ops" in result.module_sources
    assert "def add" in result.module_sources["ops"]
    assert "def double" in result.module_sources["ops"]


def test_materialized_source_parses(graph):
    mat = Materializer()
    result = mat.materialize(graph)
    for module_name, source in result.module_sources.items():
        ast.parse(source)  # should not raise


def test_round_trip_no_diffs(graph):
    mat = Materializer()
    result = mat.materialize(graph)
    parser = RoundTripParser()
    recovered = parser.parse(result.module_sources)
    diffs = parser.verify_round_trip(graph, recovered)
    assert diffs == [], f"Round-trip diffs: {diffs}"


def test_materialize_empty_graph():
    mat = Materializer()
    result = mat.materialize(GraphState())
    assert result.success is True
    assert result.module_sources == {}


def test_materialize_node_without_template():
    graph = GraphState(
        modules=[GraphModule(name="m", responsibility="x")],
        nodes=[GraphNode(
            name="stub",
            module="m",
            decl_order=0,
            signature=NodeSignature(params=[], return_type="None"),
        )],
        edges=[],
    )
    mat = Materializer()
    result = mat.materialize(graph)
    assert result.success is True
    assert "raise NotImplementedError" in result.module_sources["m"]

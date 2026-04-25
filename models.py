"""Pydantic models for the GraphForge OpenENV environment.

Complete type hierarchy for graph state, actions, observations, and task specs.
Includes models for both the original GraphForge graph-construction environment
and the AST-query episode environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from openenv.core import Action, Observation, State

# ---------------------------------------------------------------------------
# Core graph schema
# ---------------------------------------------------------------------------

ACTION_TYPES = Literal[
    "add_module",
    "remove_module",
    "add_node",
    "remove_node",
    "set_node_module",
    "attach_body",
    "add_edge",
    "remove_edge",
    "query_spec",
    "query_subgraph",
    "query_types",
    "materialize_and_validate",
    "run_behavioral_tests",
    "submit",
]

ALL_ACTION_TYPES: List[str] = [
    "add_module",
    "remove_module",
    "add_node",
    "remove_node",
    "set_node_module",
    "attach_body",
    "add_edge",
    "remove_edge",
    "query_spec",
    "query_subgraph",
    "query_types",
    "materialize_and_validate",
    "run_behavioral_tests",
    "submit",
]


class GraphModule(BaseModel):
    """A declared module (file) in the graph."""

    name: str
    responsibility: str


class ParamSpec(BaseModel):
    """A single parameter in a function signature."""

    name: str
    type_annotation: str
    default: Optional[str] = None


class NodeSignature(BaseModel):
    """Parsed function signature."""

    params: List[ParamSpec] = Field(default_factory=list)
    return_type: str = "None"


class GraphNode(BaseModel):
    """A function node in the call graph."""

    name: str
    module: str
    signature: NodeSignature
    body_template: Optional[str] = None
    body_template_args: Dict[str, Any] = Field(default_factory=dict)
    purity: Literal["pure", "impure", "unknown"] = "unknown"
    error_policy: Literal["raise", "return_none", "return_error"] = "raise"
    decl_order: int = 0


class EdgeArgMapping(BaseModel):
    """Maps one caller argument to one callee parameter."""

    caller_arg: str
    callee_param: str


class GraphEdge(BaseModel):
    """A directed call edge between two qualified function names (module.func)."""

    caller: str  # "module.func"
    callee: str  # "module.func"
    arg_mapping: List[EdgeArgMapping] = Field(default_factory=list)


class GraphState(BaseModel):
    """Complete state of the function-call graph."""

    modules: List[GraphModule] = Field(default_factory=list)
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result and supporting types
# ---------------------------------------------------------------------------


class ActionResult(BaseModel):
    """Outcome of dispatching one action."""

    success: bool
    error_kind: Optional[str] = None
    error_msg: Optional[str] = None
    query_response: Optional[Any] = None
    penalty: float = 0.0


class MaterializeResult(BaseModel):
    """Outcome of materializing the graph to Python source."""

    success: bool
    module_sources: Dict[str, str] = Field(default_factory=dict)
    parse_errors: List[str] = Field(default_factory=list)
    mypy_errors: List[str] = Field(default_factory=list)


class TestResult(BaseModel):
    """Outcome of one behavioral test."""

    test_id: str
    passed: bool
    error_msg: Optional[str] = None


class ConstraintSpec(BaseModel):
    """A single mechanically-checkable constraint."""

    constraint_id: str
    kind: str
    target: str  # dotted path e.g. "module.func" or just "module_name"
    value: Any = None
    hidden: bool = False


class ConstraintSummary(BaseModel):
    """Counts of visible constraint satisfaction."""

    total: int = 0
    visible: int = 0
    hidden: int = 0
    satisfied_visible: int = 0


class BehavioralTest(BaseModel):
    """A property-based behavioral test."""

    test_id: str
    description: str
    hypothesis_source: str  # complete pytest file as a string


class TaskSpec(BaseModel):
    """Complete specification for one episode task."""

    task_id: str
    tier: Literal[1, 2, 3]
    template_name: str
    variant: int
    description: str
    constraints: List[ConstraintSpec] = Field(default_factory=list)
    behavioral_tests: List[BehavioralTest] = Field(default_factory=list)
    token_budget: int
    turn_cap: int


# ---------------------------------------------------------------------------
# OpenENV-compliant types
# ---------------------------------------------------------------------------


class GraphObservation(Observation):
    """Observation returned by the GraphForge environment on each step.

    Extends openenv.core.Observation (adds done, reward, metadata).
    """

    episode_id: str = ""
    timestep: int = 0
    turn_budget_remaining: int = 0
    token_budget_remaining: int = 0
    graph_state: GraphState = Field(default_factory=GraphState)
    last_action_result: Optional[ActionResult] = None
    constraint_summary: ConstraintSummary = Field(default_factory=ConstraintSummary)
    visible_constraints: List[ConstraintSpec] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    task_description: str = ""


class GraphAction(Action):
    """Action sent to the GraphForge environment.

    Extends openenv.core.Action (adds metadata).
    """

    action_type: ACTION_TYPES
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GraphForgeState(State):
    """Full internal state of a GraphForge episode.

    Extends openenv.core.State (adds episode_id, step_count).
    State uses extra='allow' so additional fields are accepted.
    """

    graph: GraphState = Field(default_factory=GraphState)
    task_spec: Optional[TaskSpec] = None
    turn_cap: int = 20
    token_budget: int = 4000
    tokens_used: int = 0
    cumulative_reward: float = 0.0
    is_terminal: bool = False
    materialization_cache: Optional[Dict[str, str]] = None
    last_mypy_output: Optional[str] = None
    last_test_results: Optional[List[TestResult]] = None
    action_history: List[Tuple[str, str]] = Field(default_factory=list)
    # action_history entries: (action_type, json-serialized parameters)


# ---------------------------------------------------------------------------
# AST-query episode models
# ---------------------------------------------------------------------------

AST_ACTION_TYPES = Literal[
    "get_neighbors",
    "add_node",
    "modify_node",
    "delete_node",
    "submit",
]

ALL_AST_ACTION_TYPES: List[str] = [
    "get_neighbors",
    "add_node",
    "modify_node",
    "delete_node",
    "submit",
]


class ASTActionResult(BaseModel):
    """Outcome of one AST-episode action."""

    success: bool
    action_type: str
    message: str = ""
    added_node_ids: List[str] = Field(default_factory=list)


class ASTAction(Action):
    """Action sent to the AST-query environment."""

    action_type: AST_ACTION_TYPES
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ASTObservation(Observation):
    """Observation returned by the AST-query environment on each step."""

    episode_id: str = ""
    timestep: int = 0
    query: str = ""
    central_node: str = ""
    context_nodes: Dict[str, Any] = Field(default_factory=dict)
    context_edges: List[Dict[str, Any]] = Field(default_factory=list)
    not_found_nodes: List[str] = Field(default_factory=list)
    mutations_so_far: List[Dict[str, Any]] = Field(default_factory=list)
    last_action_result: Optional[ASTActionResult] = None
    available_actions: List[str] = Field(default_factory=list)


class ASTState(State):
    """Full internal state of one AST-query episode."""

    episode_id: str = ""
    step_count: int = 0
    query_obj: Dict[str, Any] = Field(default_factory=dict)
    graph_pkl_path: str = ""
    base_dir: str = ""
    context_nodes: Dict[str, Any] = Field(default_factory=dict)
    context_edges: List[Dict[str, Any]] = Field(default_factory=list)
    not_found_nodes: List[str] = Field(default_factory=list)
    mutations: List[Dict[str, Any]] = Field(default_factory=list)
    is_terminal: bool = False
    log_path: str = ""

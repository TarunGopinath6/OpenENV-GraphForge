"""Action dispatcher: validates and executes all 14 GraphForge actions."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionResult,
    EdgeArgMapping,
    GraphAction,
    GraphEdge,
    GraphForgeState,
    GraphModule,
    GraphNode,
    GraphState,
    NodeSignature,
)
from engine.behavioral_test_runner import BehavioralTestRunner
from engine.body_template_library import BodyTemplateLibrary, TemplateNotFoundError
from engine.constraint_checker import ConstraintChecker
from engine.materializer import Materializer
from engine.reward_engine import RewardEngine
from engine.token_counter import TokenCounter
from engine.type_engine import SignatureParseError, TypeEngine
from engine.validator import Validator


class ActionDispatcher:
    """Routes GraphAction objects to their handlers, managing rollback on failure."""

    def __init__(
        self,
        type_engine: TypeEngine,
        btl: BodyTemplateLibrary,
        materializer: Materializer,
        validator: Validator,
        test_runner: BehavioralTestRunner,
        constraint_checker: ConstraintChecker,
        reward_engine: RewardEngine,
        token_counter: TokenCounter,
    ) -> None:
        self._te = type_engine
        self._btl = btl
        self._mat = materializer
        self._val = validator
        self._tr = test_runner
        self._cc = constraint_checker
        self._re = reward_engine
        self._tc = token_counter

    def dispatch(
        self, state: GraphForgeState, action: GraphAction, is_repeat: bool = False
    ) -> Tuple[GraphForgeState, ActionResult]:
        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler is None:
            return state, ActionResult(
                success=False,
                error_kind="unknown_action",
                error_msg=f"Unknown action type: {action.action_type!r}",
                penalty=self._re.MALFORMED_ACTION,
            )

        # Deep-copy graph for rollback
        graph_snapshot = state.graph.model_copy(deep=True)

        result = handler(state, action.parameters)

        if not result.success:
            # Roll back graph mutation
            state = state.model_copy(update={"graph": graph_snapshot})

        return state, result

    # -------------------------------------------------------------------------
    # Mutation actions
    # -------------------------------------------------------------------------

    def _handle_add_module(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        responsibility = params.get("responsibility", "")
        if not name:
            return ActionResult(success=False, error_kind="missing_param", error_msg="'name' required")
        if any(m.name == name for m in state.graph.modules):
            return ActionResult(success=False, error_kind="duplicate", error_msg=f"Module {name!r} already exists")
        state.graph.modules.append(GraphModule(name=name, responsibility=responsibility))
        return ActionResult(success=True)

    def _handle_remove_module(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        if not any(m.name == name for m in state.graph.modules):
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Module {name!r} not found")
        if any(n.module == name for n in state.graph.nodes):
            return ActionResult(success=False, error_kind="has_nodes", error_msg=f"Module {name!r} still has nodes")
        state.graph.modules = [m for m in state.graph.modules if m.name != name]
        return ActionResult(success=True)

    def _handle_add_node(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        module = params.get("module", "")
        sig_str = params.get("signature", "() -> None")
        purity = params.get("purity", "unknown")
        error_policy = params.get("error_policy", "raise")

        if not name:
            return ActionResult(success=False, error_kind="missing_param", error_msg="'name' required")
        if not any(m.name == module for m in state.graph.modules):
            return ActionResult(success=False, error_kind="unknown_module", error_msg=f"Module {module!r} not found")
        if any(n.module == module and n.name == name for n in state.graph.nodes):
            return ActionResult(success=False, error_kind="duplicate", error_msg=f"Node {module}.{name} already exists")

        try:
            signature = self._te.parse_signature(sig_str)
        except SignatureParseError as e:
            return ActionResult(success=False, error_kind="bad_signature", error_msg=str(e))

        next_order = max((n.decl_order for n in state.graph.nodes if n.module == module), default=-1) + 1
        state.graph.nodes.append(
            GraphNode(
                name=name,
                module=module,
                signature=signature,
                purity=purity,  # type: ignore[arg-type]
                error_policy=error_policy,  # type: ignore[arg-type]
                decl_order=next_order,
            )
        )
        return ActionResult(success=True)

    def _handle_remove_node(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        module = params.get("module", "")
        qualified = f"{module}.{name}"

        node = next((n for n in state.graph.nodes if n.module == module and n.name == name), None)
        if node is None:
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Node {qualified!r} not found")
        if any(e.caller == qualified or e.callee == qualified for e in state.graph.edges):
            return ActionResult(success=False, error_kind="has_edges", error_msg=f"Node {qualified!r} still has edges")

        state.graph.nodes = [n for n in state.graph.nodes if not (n.module == module and n.name == name)]
        return ActionResult(success=True)

    def _handle_set_node_module(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        current_module = params.get("current_module", "")
        new_module = params.get("new_module", "")

        if not any(m.name == new_module for m in state.graph.modules):
            return ActionResult(success=False, error_kind="unknown_module", error_msg=f"Module {new_module!r} not found")
        if any(n.module == new_module and n.name == name for n in state.graph.nodes):
            return ActionResult(success=False, error_kind="duplicate", error_msg=f"Node {new_module}.{name} already exists")

        node = next((n for n in state.graph.nodes if n.module == current_module and n.name == name), None)
        if node is None:
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Node {current_module}.{name} not found")

        # Update module and fix edges
        old_qualified = f"{current_module}.{name}"
        new_qualified = f"{new_module}.{name}"
        node.module = new_module

        for edge in state.graph.edges:
            if edge.caller == old_qualified:
                edge.caller = new_qualified
            if edge.callee == old_qualified:
                edge.callee = new_qualified

        return ActionResult(success=True)

    def _handle_attach_body(self, state: GraphForgeState, params: Dict) -> ActionResult:
        name = params.get("name", "")
        module = params.get("module", "")
        template = params.get("template", "")
        args = params.get("args", {})

        node = next((n for n in state.graph.nodes if n.module == module and n.name == name), None)
        if node is None:
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Node {module}.{name} not found")

        try:
            errors = self._btl.validate_args(template, args)
        except Exception as e:
            errors = [str(e)]

        if errors:
            return ActionResult(success=False, error_kind="template_arg_error", error_msg="; ".join(errors))

        try:
            self._btl.render(template, args)  # dry-run to catch render errors
        except Exception as e:
            return ActionResult(success=False, error_kind="render_error", error_msg=str(e))

        node.body_template = template
        node.body_template_args = args
        return ActionResult(success=True)

    def _handle_add_edge(self, state: GraphForgeState, params: Dict) -> ActionResult:
        caller = params.get("caller", "")
        callee = params.get("callee", "")
        arg_mapping_raw = params.get("arg_mapping", [])

        node_map = {f"{n.module}.{n.name}": n for n in state.graph.nodes}
        if caller not in node_map:
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Caller {caller!r} not found")
        if callee not in node_map:
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Callee {callee!r} not found")

        if any(e.caller == caller and e.callee == callee for e in state.graph.edges):
            return ActionResult(success=False, error_kind="duplicate", error_msg=f"Edge {caller}->{callee} already exists")

        arg_mapping = [
            EdgeArgMapping(caller_arg=m["caller_arg"], callee_param=m["callee_param"])
            for m in arg_mapping_raw
        ]

        # Type validation
        type_errors = self._te.validate_edge_types(node_map[caller], node_map[callee], arg_mapping)
        if type_errors:
            return ActionResult(success=False, error_kind="type_mismatch", error_msg="; ".join(type_errors))

        # Cycle check
        proposed_edge = GraphEdge(caller=caller, callee=callee, arg_mapping=arg_mapping)
        if self._te.has_import_cycle(state.graph, proposed_edge):
            return ActionResult(success=False, error_kind="import_cycle", error_msg="Adding this edge would create an import cycle")

        state.graph.edges.append(proposed_edge)
        return ActionResult(success=True)

    def _handle_remove_edge(self, state: GraphForgeState, params: Dict) -> ActionResult:
        caller = params.get("caller", "")
        callee = params.get("callee", "")

        if not any(e.caller == caller and e.callee == callee for e in state.graph.edges):
            return ActionResult(success=False, error_kind="not_found", error_msg=f"Edge {caller}->{callee} not found")

        state.graph.edges = [e for e in state.graph.edges if not (e.caller == caller and e.callee == callee)]
        return ActionResult(success=True)

    # -------------------------------------------------------------------------
    # Information actions
    # -------------------------------------------------------------------------

    def _handle_query_spec(self, state: GraphForgeState, params: Dict) -> ActionResult:
        if state.task_spec is None:
            return ActionResult(success=True, query_response={"total_visible": 0, "satisfied": 0, "unsatisfied": 0, "visible_constraints": [], "summary": {}})

        constraint_kind = params.get("constraint_kind")
        visible = [c for c in state.task_spec.constraints if not c.hidden]

        if constraint_kind:
            visible = [c for c in visible if c.kind == constraint_kind]

        results = self._cc.check_all(state.graph, visible)
        satisfied = sum(results.values())

        return ActionResult(
            success=True,
            query_response={
                "total_visible": len(visible),
                "satisfied": satisfied,
                "unsatisfied": len(visible) - satisfied,
                "constraints": [
                    {
                        "id": c.constraint_id,
                        "kind": c.kind,
                        "target": c.target,
                        "satisfied": results[c.constraint_id],
                    }
                    for c in visible
                ],
            },
        )

    def _handle_query_subgraph(self, state: GraphForgeState, params: Dict) -> ActionResult:
        scope = params.get("scope", "")

        if scope.startswith("module:"):
            mod_name = scope[len("module:"):]
            nodes = [n for n in state.graph.nodes if n.module == mod_name]
            edges = [
                e for e in state.graph.edges
                if e.caller.startswith(f"{mod_name}.") or e.callee.startswith(f"{mod_name}.")
            ]
            return ActionResult(
                success=True,
                query_response={
                    "scope": scope,
                    "nodes": [f"{n.module}.{n.name}" for n in nodes],
                    "edges": [f"{e.caller}->{e.callee}" for e in edges],
                },
            )
        elif scope.startswith("neighbors:"):
            node_name = scope[len("neighbors:"):]
            callers = [e.caller for e in state.graph.edges if e.callee == node_name]
            callees = [e.callee for e in state.graph.edges if e.caller == node_name]
            return ActionResult(
                success=True,
                query_response={"node": node_name, "callers": callers, "callees": callees},
            )
        elif scope.startswith("path:"):
            parts = scope[len("path:"):].split(":", 1)
            if len(parts) == 2:
                path = self._find_path(state.graph, parts[0], parts[1])
                return ActionResult(success=True, query_response={"path": path})

        return ActionResult(success=False, error_kind="invalid_scope", error_msg=f"Unrecognized scope: {scope!r}")

    def _handle_query_types(self, state: GraphForgeState, params: Dict) -> ActionResult:
        scope = params.get("scope", "all")
        nodes = state.graph.nodes

        if scope != "all":
            mod = scope if "." not in scope else scope.split(".")[0]
            nodes = [n for n in nodes if n.module == mod or f"{n.module}.{n.name}" == scope]

        type_info = {}
        for n in nodes:
            key = f"{n.module}.{n.name}"
            params_info = {p.name: p.type_annotation for p in n.signature.params}
            has_any = self._te.detect_any_contamination(n)
            type_info[key] = {
                "params": params_info,
                "return_type": n.signature.return_type,
                "has_any": has_any,
            }

        type_errors = self._te.check_type_consistency(state.graph)

        return ActionResult(
            success=True,
            query_response={
                "nodes": type_info,
                "type_errors": type_errors,
            },
        )

    def _handle_materialize_and_validate(self, state: GraphForgeState, params: Dict) -> ActionResult:
        mat_result = self._mat.materialize(state.graph)
        if not mat_result.success:
            state.materialization_cache = None
            return ActionResult(
                success=False,
                error_kind="materialization_failed",
                error_msg="; ".join(mat_result.parse_errors),
                query_response=mat_result.model_dump(),
            )

        val_result = self._val.full_validate(mat_result.module_sources)
        state.materialization_cache = mat_result.module_sources
        state.last_mypy_output = "; ".join(val_result.mypy_errors) if val_result.mypy_errors else ""

        return ActionResult(
            success=True,
            query_response={
                "materialized": True,
                "parse_errors": val_result.parse_errors,
                "mypy_ok": len(val_result.mypy_errors) == 0,
                "mypy_errors": val_result.mypy_errors,
            },
        )

    def _handle_run_behavioral_tests(self, state: GraphForgeState, params: Dict) -> ActionResult:
        if state.materialization_cache is None:
            return ActionResult(
                success=False,
                error_kind="no_materialization",
                error_msg="Call materialize_and_validate first",
            )
        if state.task_spec is None:
            return ActionResult(success=True, query_response={"results": []})

        results = self._tr.run_tests(
            state.materialization_cache,
            state.task_spec.behavioral_tests,
        )
        state.last_test_results = results
        return ActionResult(
            success=True,
            query_response={
                "results": [r.model_dump() for r in results],
                "passed": sum(1 for r in results if r.passed),
                "total": len(results),
            },
        )

    def _handle_submit(self, state: GraphForgeState, params: Dict) -> ActionResult:
        # Materialize if needed
        if state.materialization_cache is None:
            mat_result = self._mat.materialize(state.graph)
            if not mat_result.success:
                state.is_terminal = True
                return ActionResult(
                    success=False,
                    error_kind="materialization_failed",
                    error_msg="Cannot submit: graph fails to materialize",
                    penalty=self._re.MATERIALIZE_FAIL_PENALTY,
                    query_response={"terminal_reward": self._re.MATERIALIZE_FAIL_PENALTY},
                )
            state.materialization_cache = mat_result.module_sources

        # Run mypy
        val_result = self._val.full_validate(state.materialization_cache)
        mypy_ok = len(val_result.mypy_errors) == 0

        # Run behavioral tests
        test_results = state.last_test_results or []
        if state.task_spec and state.task_spec.behavioral_tests:
            test_results = self._tr.run_tests(
                state.materialization_cache,
                state.task_spec.behavioral_tests,
            )
            state.last_test_results = test_results

        # Compute terminal reward against all (including hidden) constraints
        constraints = state.task_spec.constraints if state.task_spec else []
        mat_ok_result = self._mat.materialize(state.graph)

        terminal_reward = self._re.terminal_reward(
            graph=state.graph,
            constraints=constraints,
            test_results=test_results,
            materialize_result=mat_ok_result,
            mypy_ok=mypy_ok,
            tokens_used=state.tokens_used,
            token_budget=state.token_budget,
        )

        state.is_terminal = True
        state.cumulative_reward += terminal_reward

        return ActionResult(
            success=True,
            query_response={
                "terminal_reward": terminal_reward,
                "mypy_ok": mypy_ok,
                "tests_passed": sum(1 for r in test_results if r.passed),
                "tests_total": len(test_results),
            },
            penalty=0.0,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _find_path(self, graph: GraphState, start: str, end: str) -> Optional[List[str]]:
        adj: Dict[str, List[str]] = {}
        for edge in graph.edges:
            adj.setdefault(edge.caller, []).append(edge.callee)

        from collections import deque
        queue: deque = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        return None

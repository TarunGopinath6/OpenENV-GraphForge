"""Constraint checker: evaluates all constraint kinds against a GraphState."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from models import ConstraintSpec, GraphState


class ConstraintChecker:
    """Mechanically checks all constraint kinds defined in the spec."""

    def check_all(
        self, graph: GraphState, constraints: List[ConstraintSpec]
    ) -> Dict[str, bool]:
        """Return {constraint_id: satisfied} for every constraint."""
        return {c.constraint_id: self.check_one(graph, c) for c in constraints}

    def check_one(self, graph: GraphState, spec: ConstraintSpec) -> bool:
        handler = getattr(self, f"_check_{spec.kind}", None)
        if handler is None:
            return False
        try:
            return bool(handler(graph, spec.target, spec.value))
        except Exception:
            return False

    def count_satisfied(
        self, graph: GraphState, constraints: List[ConstraintSpec]
    ) -> Tuple[int, int]:
        """Return (satisfied_count, total_count)."""
        results = self.check_all(graph, constraints)
        return sum(results.values()), len(results)

    # -------------------------------------------------------------------------
    # Structural constraints
    # -------------------------------------------------------------------------

    def _check_node_exists(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func_name" or just "func_name" """
        return self._resolve_node(graph, target) is not None

    def _check_node_absent(self, graph: GraphState, target: str, value) -> bool:
        return self._resolve_node(graph, target) is None

    def _check_edge_exists(self, graph: GraphState, target: str, value) -> bool:
        """target = "caller_module.caller::callee_module.callee" or value holds callee"""
        caller, callee = self._parse_edge_target(target, value)
        return any(e.caller == caller and e.callee == callee for e in graph.edges)

    def _check_edge_absent(self, graph: GraphState, target: str, value) -> bool:
        caller, callee = self._parse_edge_target(target, value)
        return not any(e.caller == caller and e.callee == callee for e in graph.edges)

    def _check_module_count(self, graph: GraphState, target: str, value) -> bool:
        return len(graph.modules) == int(value)

    def _check_module_count_range(self, graph: GraphState, target: str, value) -> bool:
        lo, hi = value  # [min, max]
        return int(lo) <= len(graph.modules) <= int(hi)

    def _check_module_size_max(self, graph: GraphState, target: str, value) -> bool:
        """target = module name, value = max node count"""
        count = sum(1 for n in graph.nodes if n.module == target)
        return count <= int(value)

    def _check_module_responsibility(self, graph: GraphState, target: str, value) -> bool:
        """target = module name, value = responsibility tag"""
        return any(m.name == target and m.responsibility == value for m in graph.modules)

    def _check_acyclic_imports(self, graph: GraphState, target: str, value) -> bool:
        from engine.type_engine import TypeEngine
        te = TypeEngine()
        adj = te.build_import_graph(graph)
        return not te._has_cycle(adj)

    def _check_entrypoint_exists(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func". The function must exist."""
        return self._resolve_node(graph, target) is not None

    def _check_internal_only(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func". True if all callers are in the same module."""
        mod = target.split(".")[0] if "." in target else None
        for edge in graph.edges:
            if edge.callee == target:
                caller_mod = edge.caller.split(".")[0]
                if caller_mod != mod:
                    return False
        return True

    def _check_fan_in_max(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func", value = max callers"""
        count = sum(1 for e in graph.edges if e.callee == target)
        return count <= int(value)

    def _check_fan_out_max(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func", value = max callees"""
        count = sum(1 for e in graph.edges if e.caller == target)
        return count <= int(value)

    def _check_dag_depth_max(self, graph: GraphState, target: str, value) -> bool:
        """target ignored (whole-graph). value = max depth of longest path."""
        max_depth = int(value)
        return self._compute_dag_depth(graph) <= max_depth

    # -------------------------------------------------------------------------
    # Type / signature constraints
    # -------------------------------------------------------------------------

    def _check_signature_matches(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func", value = regex pattern for signature string"""
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        sig_str = self._node_signature_str(node)
        return bool(re.search(str(value), sig_str))

    def _check_return_type(self, graph: GraphState, target: str, value) -> bool:
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        return node.signature.return_type == str(value)

    def _check_arg_type(self, graph: GraphState, target: str, value) -> bool:
        """target = "module.func", value = [position_or_name, type_str]"""
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        pos_or_name, expected_type = value
        params = node.signature.params
        if isinstance(pos_or_name, int):
            if pos_or_name >= len(params):
                return False
            return params[pos_or_name].type_annotation == expected_type
        return any(p.name == pos_or_name and p.type_annotation == expected_type for p in params)

    def _check_type_consistency(self, graph: GraphState, target: str, value) -> bool:
        """Check all edges are type-compatible."""
        from engine.type_engine import TypeEngine
        te = TypeEngine()
        return len(te.check_type_consistency(graph)) == 0

    def _check_no_any_types(self, graph: GraphState, target: str, value) -> bool:
        from engine.type_engine import TypeEngine
        te = TypeEngine()
        return not any(te.detect_any_contamination(n) for n in graph.nodes)

    def _check_pure_function(self, graph: GraphState, target: str, value) -> bool:
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        return node.purity == "pure"

    # -------------------------------------------------------------------------
    # Behavioral / materialization constraints (graph-level only; full checks
    # require running external tools and are handled by ActionDispatcher)
    # -------------------------------------------------------------------------

    def _check_materializes(self, graph: GraphState, target: str, value) -> bool:
        """Only passable if materialization_cache is set — caller must inject."""
        # Resolved by ActionDispatcher/environment using cached result
        return False  # conservative; override when cache is available

    def _check_imports_resolve(self, graph: GraphState, target: str, value) -> bool:
        return False

    def _check_type_checks(self, graph: GraphState, target: str, value) -> bool:
        return False

    def _check_behavioral_test_passes(self, graph: GraphState, target: str, value) -> bool:
        return False

    def _check_error_handling_present(self, graph: GraphState, target: str, value) -> bool:
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        return node.error_policy in ("return_none", "return_error")

    def _check_error_handling_absent(self, graph: GraphState, target: str, value) -> bool:
        node = self._resolve_node(graph, target)
        if node is None:
            return False
        return node.error_policy == "raise"

    # -------------------------------------------------------------------------
    # Extended check with cached materialization results
    # -------------------------------------------------------------------------

    def check_all_with_cache(
        self,
        graph: GraphState,
        constraints: List[ConstraintSpec],
        materialize_ok: bool = False,
        imports_ok: bool = False,
        mypy_ok: bool = False,
        test_results: Dict[str, bool] | None = None,
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for c in constraints:
            if c.kind == "materializes":
                results[c.constraint_id] = materialize_ok
            elif c.kind == "imports_resolve":
                results[c.constraint_id] = imports_ok
            elif c.kind == "type_checks":
                results[c.constraint_id] = mypy_ok
            elif c.kind == "behavioral_test_passes":
                if test_results is not None:
                    results[c.constraint_id] = test_results.get(str(c.value), False)
                else:
                    results[c.constraint_id] = False
            else:
                results[c.constraint_id] = self.check_one(graph, c)
        return results

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_node(self, graph: GraphState, target: str):
        """Return the node for "module.func" or "func" (first match)."""
        if "." in target:
            module, name = target.split(".", 1)
            for n in graph.nodes:
                if n.module == module and n.name == name:
                    return n
        else:
            for n in graph.nodes:
                if n.name == target:
                    return n
        return None

    def _parse_edge_target(self, target: str, value) -> Tuple[str, str]:
        """Parse edge spec from target+value or target alone (caller::callee format)."""
        if "::" in target:
            parts = target.split("::", 1)
            return parts[0].strip(), parts[1].strip()
        if value is not None:
            return str(target), str(value)
        # fallback
        return str(target), ""

    def _node_signature_str(self, node) -> str:
        params = ", ".join(
            f"{p.name}: {p.type_annotation}" for p in node.signature.params
        )
        return f"({params}) -> {node.signature.return_type}"

    def _compute_dag_depth(self, graph: GraphState) -> int:
        """Return the length of the longest path in the call graph."""
        nodes = {f"{n.module}.{n.name}" for n in graph.nodes}
        adjacency: Dict[str, Set[str]] = {n: set() for n in nodes}
        for edge in graph.edges:
            if edge.caller in adjacency:
                adjacency[edge.caller].add(edge.callee)

        memo: Dict[str, int] = {}

        def depth(n: str) -> int:
            if n in memo:
                return memo[n]
            children = adjacency.get(n, set())
            result = 0 if not children else 1 + max(depth(c) for c in children)
            memo[n] = result
            return result

        if not nodes:
            return 0
        return max(depth(n) for n in nodes)

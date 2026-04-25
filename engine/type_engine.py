"""Type engine: signature parsing, edge type validation, import cycle detection."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from models import EdgeArgMapping, GraphEdge, GraphNode, GraphState, NodeSignature, ParamSpec


class SignatureParseError(ValueError):
    pass


# Type pairs considered compatible (one-way)
_COMPATIBLE_PAIRS: Set[Tuple[str, str]] = {
    ("int", "float"),
    ("int", "complex"),
    ("float", "complex"),
    ("bool", "int"),
}


class TypeEngine:
    """Handles all type-related operations on the graph."""

    # ---------------------------------------------------------------------------
    # Signature parsing
    # ---------------------------------------------------------------------------

    def parse_signature(self, sig_str: str) -> NodeSignature:
        """Parse a Python function signature string into NodeSignature.

        sig_str should be in the form: (param: type, ...) -> return_type
        """
        stub = f"def _f{sig_str}: pass"
        try:
            tree = ast.parse(stub, mode="exec")
        except SyntaxError as e:
            raise SignatureParseError(f"Invalid signature {sig_str!r}: {e}") from e

        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise SignatureParseError(f"Could not parse signature: {sig_str!r}")

        args = func_def.args
        params: List[ParamSpec] = []

        all_args = args.posonlyargs + args.args
        defaults_offset = len(all_args) - len(args.defaults)

        for i, arg in enumerate(all_args):
            ann = self._annotation_to_str(arg.annotation) if arg.annotation else "Any"
            default: Optional[str] = None
            if i >= defaults_offset:
                d = args.defaults[i - defaults_offset]
                default = ast.unparse(d)
            params.append(ParamSpec(name=arg.arg, type_annotation=ann, default=default))

        if args.vararg:
            ann = self._annotation_to_str(args.vararg.annotation) if args.vararg.annotation else "Any"
            params.append(ParamSpec(name=f"*{args.vararg.arg}", type_annotation=ann))

        if args.kwarg:
            ann = self._annotation_to_str(args.kwarg.annotation) if args.kwarg.annotation else "Any"
            params.append(ParamSpec(name=f"**{args.kwarg.arg}", type_annotation=ann))

        return_type = "None"
        if func_def.returns:
            return_type = self._annotation_to_str(func_def.returns)

        return NodeSignature(params=params, return_type=return_type)

    def _annotation_to_str(self, node: Optional[ast.expr]) -> str:
        if node is None:
            return "Any"
        return ast.unparse(node)

    # ---------------------------------------------------------------------------
    # Edge type validation
    # ---------------------------------------------------------------------------

    def validate_edge_types(
        self,
        caller_node: GraphNode,
        callee_node: GraphNode,
        arg_mapping: List[EdgeArgMapping],
    ) -> List[str]:
        """Return a list of type-mismatch error strings (empty = valid)."""
        errors: List[str] = []

        caller_params = {p.name: p.type_annotation for p in caller_node.signature.params}
        callee_params = {p.name: p.type_annotation for p in callee_node.signature.params}

        for mapping in arg_mapping:
            # caller_arg must exist in caller params
            if mapping.caller_arg not in caller_params:
                errors.append(
                    f"Caller param {mapping.caller_arg!r} not in {caller_node.name}'s signature"
                )
                continue
            # callee_param must exist in callee params
            if mapping.callee_param not in callee_params:
                errors.append(
                    f"Callee param {mapping.callee_param!r} not in {callee_node.name}'s signature"
                )
                continue

            src_type = caller_params[mapping.caller_arg]
            dst_type = callee_params[mapping.callee_param]
            if not self._are_compatible(src_type, dst_type):
                errors.append(
                    f"Type mismatch: {caller_node.name}.{mapping.caller_arg}: {src_type} "
                    f"→ {callee_node.name}.{mapping.callee_param}: {dst_type}"
                )

        return errors

    def _are_compatible(self, src: str, dst: str) -> bool:
        if src == dst:
            return True
        if src == "Any" or dst == "Any":
            return True
        if (src, dst) in _COMPATIBLE_PAIRS:
            return True
        return False

    # ---------------------------------------------------------------------------
    # Any contamination
    # ---------------------------------------------------------------------------

    def detect_any_contamination(self, node: GraphNode) -> bool:
        """True if any param or return type is Any or unannotated."""
        for p in node.signature.params:
            if p.type_annotation in ("Any", ""):
                return True
        if node.signature.return_type in ("Any", ""):
            return True
        return False

    # ---------------------------------------------------------------------------
    # Import graph and cycle detection
    # ---------------------------------------------------------------------------

    def build_import_graph(self, graph: GraphState) -> Dict[str, Set[str]]:
        """Return adjacency dict: module -> set of modules it imports."""
        modules = {m.name for m in graph.modules}
        adj: Dict[str, Set[str]] = {m: set() for m in modules}
        for edge in graph.edges:
            caller_mod = edge.caller.split(".")[0]
            callee_mod = edge.callee.split(".")[0]
            if caller_mod != callee_mod and caller_mod in adj:
                adj[caller_mod].add(callee_mod)
        return adj

    def has_import_cycle(self, graph: GraphState, proposed_edge: GraphEdge) -> bool:
        """True if adding proposed_edge would create a cycle in the import graph."""
        caller_mod = proposed_edge.caller.split(".")[0]
        callee_mod = proposed_edge.callee.split(".")[0]

        if caller_mod == callee_mod:
            return False

        adj = self.build_import_graph(graph)
        # Tentatively add the edge
        if caller_mod not in adj:
            adj[caller_mod] = set()
        adj[caller_mod].add(callee_mod)

        return self._has_cycle(adj)

    def _has_cycle(self, adj: Dict[str, Set[str]]) -> bool:
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    if dfs(nb):
                        return True
                elif nb in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for n in list(adj):
            if n not in visited:
                if dfs(n):
                    return True
        return False

    # ---------------------------------------------------------------------------
    # Helpers used by ConstraintChecker
    # ---------------------------------------------------------------------------

    def check_type_consistency(self, graph: GraphState) -> List[str]:
        """Check all edges for type compatibility. Returns list of error strings."""
        node_map = {f"{n.module}.{n.name}": n for n in graph.nodes}
        errors: List[str] = []
        for edge in graph.edges:
            caller_node = node_map.get(edge.caller)
            callee_node = node_map.get(edge.callee)
            if caller_node and callee_node:
                errs = self.validate_edge_types(caller_node, callee_node, edge.arg_mapping)
                errors.extend(errs)
        return errors

"""Round-trip parser: recovers a GraphState from materialized Python source."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from models import (
    EdgeArgMapping,
    GraphEdge,
    GraphModule,
    GraphNode,
    GraphState,
    NodeSignature,
    ParamSpec,
)


class RoundTripParser:
    """AST-based parser that recovers GraphState from materializer output."""

    def parse(self, module_sources: Dict[str, str]) -> GraphState:
        all_modules: List[GraphModule] = []
        all_nodes: List[GraphNode] = []
        all_edges: List[GraphEdge] = []

        for module_name, source in module_sources.items():
            module, nodes, edges = self.parse_module(module_name, source)
            all_modules.append(module)
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        return GraphState(modules=all_modules, nodes=all_nodes, edges=all_edges)

    def parse_module(
        self, module_name: str, source: str
    ) -> Tuple[GraphModule, List[GraphNode], List[GraphEdge]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return GraphModule(name=module_name, responsibility="unknown"), [], []

        # Extract responsibility from module docstring
        responsibility = "unknown"
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            docstring = tree.body[0].value.value
            # Format: "module_name: responsibility"
            if ":" in docstring:
                responsibility = docstring.split(":", 1)[1].strip()
            else:
                responsibility = docstring.strip()

        module = GraphModule(name=module_name, responsibility=responsibility)

        # Collect imports from this module (for edge recovery)
        imported_from: Dict[str, str] = {}  # func_name -> source_module
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_from[name] = node.module

        # Parse function definitions
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        decl_order = 0

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            graph_node = self._parse_function(module_name, node, decl_order)
            nodes.append(graph_node)
            decl_order += 1

            # Recover call edges from function body
            for call_edge in self._extract_call_edges(
                module_name, node.name, node, imported_from
            ):
                edges.append(call_edge)

        return module, nodes, edges

    def _parse_function(
        self, module_name: str, func_def: ast.FunctionDef, decl_order: int
    ) -> GraphNode:
        signature = self._parse_function_signature(func_def)
        return GraphNode(
            name=func_def.name,
            module=module_name,
            signature=signature,
            decl_order=decl_order,
        )

    def _parse_function_signature(self, func_def: ast.FunctionDef) -> NodeSignature:
        args = func_def.args
        all_args = args.posonlyargs + args.args
        defaults_offset = len(all_args) - len(args.defaults)

        params: List[ParamSpec] = []
        for i, arg in enumerate(all_args):
            ann = ast.unparse(arg.annotation) if arg.annotation else "Any"
            default: Optional[str] = None
            if i >= defaults_offset:
                default = ast.unparse(args.defaults[i - defaults_offset])
            params.append(ParamSpec(name=arg.arg, type_annotation=ann, default=default))

        return_type = "None"
        if func_def.returns:
            return_type = ast.unparse(func_def.returns)

        return NodeSignature(params=params, return_type=return_type)

    def _extract_call_edges(
        self,
        module_name: str,
        func_name: str,
        func_def: ast.FunctionDef,
        imported_from: Dict[str, str],
    ) -> List[GraphEdge]:
        caller = f"{module_name}.{func_name}"
        edges: List[GraphEdge] = []
        seen: set = set()

        for node in ast.walk(func_def):
            if not isinstance(node, ast.Call):
                continue
            callee_name: Optional[str] = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr

            if callee_name and callee_name in imported_from:
                callee_module = imported_from[callee_name]
                callee = f"{callee_module}.{callee_name}"
                if callee not in seen:
                    seen.add(callee)
                    edges.append(GraphEdge(caller=caller, callee=callee, arg_mapping=[]))

        return edges

    def verify_round_trip(self, original: GraphState, recovered: GraphState) -> List[str]:
        """Return list of discrepancies between original and recovered states."""
        diffs: List[str] = []

        orig_mods = {m.name for m in original.modules}
        rec_mods = {m.name for m in recovered.modules}
        for m in orig_mods - rec_mods:
            diffs.append(f"Missing module in recovered: {m!r}")
        for m in rec_mods - orig_mods:
            diffs.append(f"Extra module in recovered: {m!r}")

        orig_nodes = {f"{n.module}.{n.name}" for n in original.nodes}
        rec_nodes = {f"{n.module}.{n.name}" for n in recovered.nodes}
        for n in orig_nodes - rec_nodes:
            diffs.append(f"Missing node in recovered: {n!r}")
        for n in rec_nodes - orig_nodes:
            diffs.append(f"Extra node in recovered: {n!r}")

        # Check return types match
        orig_node_map = {f"{n.module}.{n.name}": n for n in original.nodes}
        rec_node_map = {f"{n.module}.{n.name}": n for n in recovered.nodes}
        for key in orig_nodes & rec_nodes:
            orig_rt = orig_node_map[key].signature.return_type
            rec_rt = rec_node_map[key].signature.return_type
            if orig_rt != rec_rt:
                diffs.append(f"Node {key!r} return type: {orig_rt!r} vs {rec_rt!r}")

        return diffs

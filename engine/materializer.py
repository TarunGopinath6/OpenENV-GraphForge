"""Materializer: converts a GraphState to Python source files."""

from __future__ import annotations

import textwrap
from typing import Dict, List

from models import GraphEdge, GraphNode, GraphState, MaterializeResult
from engine.body_template_library import BodyTemplateLibrary


class Materializer:
    """Projects a GraphState to a dict of {module_name: python_source}."""

    def __init__(self) -> None:
        self._btl = BodyTemplateLibrary()

    def materialize(self, graph: GraphState) -> MaterializeResult:
        module_sources: Dict[str, str] = {}
        parse_errors: List[str] = []

        for module in graph.modules:
            nodes = [n for n in graph.nodes if n.module == module.name]
            edges = list(graph.edges)
            try:
                source = self._build_module_source(module.name, module.responsibility, nodes, edges)
            except Exception as e:
                parse_errors.append(f"Module {module.name!r}: {e}")
                source = f'# ERROR generating module {module.name}: {e}\n'
            module_sources[module.name] = source

        success = len(parse_errors) == 0
        return MaterializeResult(
            success=success,
            module_sources=module_sources,
            parse_errors=parse_errors,
            mypy_errors=[],
        )

    def _build_module_source(
        self,
        module_name: str,
        responsibility: str,
        nodes: List[GraphNode],
        all_edges: List[GraphEdge],
    ) -> str:
        lines: List[str] = []

        # Module docstring
        lines.append(f'"""{module_name}: {responsibility}"""')
        lines.append("")

        # Imports: from <other_module> import <func>
        imports = self._compute_imports(module_name, nodes, all_edges)
        if imports:
            for imp in sorted(imports):
                lines.append(imp)
            lines.append("")

        # Functions in decl_order
        sorted_nodes = self._sort_by_decl_order(nodes)
        for node in sorted_nodes:
            lines.extend(self._generate_function_source(node))
            lines.append("")

        return "\n".join(lines)

    def _compute_imports(
        self,
        module_name: str,
        nodes: List[GraphNode],
        all_edges: List[GraphEdge],
    ) -> List[str]:
        node_names = {n.name for n in nodes}
        imports: Dict[str, set] = {}  # callee_module -> set of callee_func

        for edge in all_edges:
            caller_mod = edge.caller.split(".")[0]
            callee_parts = edge.callee.split(".", 1)
            if len(callee_parts) < 2:
                continue
            callee_mod, callee_func = callee_parts
            if caller_mod == module_name and callee_mod != module_name:
                imports.setdefault(callee_mod, set()).add(callee_func)

        result: List[str] = []
        for mod in sorted(imports):
            funcs = ", ".join(sorted(imports[mod]))
            result.append(f"from {mod} import {funcs}")
        return result

    def _sort_by_decl_order(self, nodes: List[GraphNode]) -> List[GraphNode]:
        return sorted(nodes, key=lambda n: n.decl_order)

    def _generate_function_source(self, node: GraphNode) -> List[str]:
        # Build signature line
        params = ", ".join(
            f"{p.name}: {p.type_annotation}" + (f" = {p.default}" if p.default else "")
            for p in node.signature.params
        )
        sig_line = f"def {node.name}({params}) -> {node.signature.return_type}:"
        lines = [sig_line]

        # Body
        if node.body_template:
            try:
                body_src = self._btl.render(node.body_template, node.body_template_args)
                for body_line in body_src.splitlines():
                    lines.append(f"    {body_line}")
            except Exception as e:
                lines.append(f"    raise NotImplementedError({str(e)!r})")
        else:
            lines.append("    raise NotImplementedError")

        return lines

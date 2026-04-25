import ast
import os
import networkx as nx

class CodebaseDAGBuilder:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graph = nx.DiGraph()
        self.current_file = None
        self.current_scope = []

    def build(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self.process_file(filepath)
        return self.graph

    def process_file(self, filepath):
        self.current_file = filepath
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
                self.visit(tree)
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

    def add_node(self, name, node_type):
        self.graph.add_node(name, type=node_type)

    def add_edge(self, src, dst, relation):
        self.graph.add_edge(src, dst, relation=relation)

    def get_current_scope_name(self):
        return "::".join(self.current_scope) if self.current_scope else self.current_file

    def visit(self, node):
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    # -------- Nodes --------

    def visit_Module(self, node):
        module_name = self.current_file
        self.add_node(module_name, "module")
        self.current_scope.append(module_name)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_ClassDef(self, node):
        class_name = f"{self.get_current_scope_name()}::{node.name}"
        self.add_node(class_name, "class")
        self.add_edge(self.get_current_scope_name(), class_name, "contains")

        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_FunctionDef(self, node):
        func_name = f"{self.get_current_scope_name()}::{node.name}"
        self.add_node(func_name, "function")
        self.add_edge(self.get_current_scope_name(), func_name, "contains")

        self.current_scope.append(node.name)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    # -------- Relationships --------

    def visit_Call(self, node):
        caller = self.get_current_scope_name()

        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        else:
            callee = "unknown_call"

        self.add_node(callee, "external_or_function")
        self.add_edge(caller, callee, "calls")

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.add_node(alias.name, "module")
            self.add_edge(self.get_current_scope_name(), alias.name, "imports")

    def visit_ImportFrom(self, node):
        module = node.module or "unknown"
        for alias in node.names:
            imported = f"{module}.{alias.name}"
            self.add_node(imported, "module")
            self.add_edge(self.get_current_scope_name(), imported, "imports")
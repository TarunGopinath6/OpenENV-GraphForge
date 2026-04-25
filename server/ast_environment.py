"""AST-query OpenENV environment — iterative graph reasoning and mutation."""

from __future__ import annotations

import ast as _ast
import json
import os
import pickle
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from openenv.core import Environment

from models import (
    ALL_AST_ACTION_TYPES,
    ASTAction,
    ASTActionResult,
    ASTObservation,
    ASTState,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_DIR = os.path.join(_PROJECT_ROOT, "sample-codebase")

MAX_STEPS = 50


# ---------------------------------------------------------------------------
# Graph utility functions (adapted from sample-codebase/answer_queries.py)
# ---------------------------------------------------------------------------

def _get_node_source(node_id: str, base_dir: str) -> Optional[str]:
    """Extract source lines for a '::'-delimited node ID, or None."""
    if "::" not in node_id:
        return None
    file_part, symbol = node_id.rsplit("::", 1)
    file_path = os.path.join(base_dir, file_part)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = _ast.parse(source)
        lines = source.splitlines()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                if node.name == symbol:
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    except Exception:
        pass
    return None


def _get_neighbors(
    G: nx.DiGraph, node_id: str, base_dir: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (nodes_dict, edges_list) for node_id and all its neighbors."""
    if node_id not in G:
        return {}, []
    neighbors = list(G.successors(node_id)) + list(G.predecessors(node_id))
    new_nodes: Dict[str, Any] = {}
    new_edges: List[Dict[str, Any]] = []
    for n in neighbors + [node_id]:
        data: Dict[str, Any] = {"type": G.nodes[n].get("type", "unknown")}
        src = _get_node_source(n, base_dir)
        if src:
            data["source"] = src
        new_nodes[n] = data
    for u, v in G.edges(neighbors + [node_id]):
        if u in new_nodes and v in new_nodes:
            new_edges.append({"source": u, "target": v, "attributes": dict(G.edges[u, v])})
    return new_nodes, new_edges


def _normalize_subgraph(
    subgraph: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize node/edge formats across the different query JSON files."""
    nodes_raw = subgraph.get("nodes", {})
    nodes: Dict[str, Any] = (
        {nid: {} for nid in nodes_raw} if isinstance(nodes_raw, list) else dict(nodes_raw)
    )
    edges: List[Dict[str, Any]] = []
    for e in subgraph.get("edges", []):
        if isinstance(e, (list, tuple)):
            edges.append({"source": e[0], "target": e[1], "attributes": {}})
        else:
            edges.append(dict(e))
    return nodes, edges


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ASTEnvironment(Environment[ASTAction, ASTObservation, ASTState]):
    """OpenENV environment for LLM-driven AST reasoning and code mutation.

    On each reset() a query is chosen at random from sample-codebase/queries/.
    The LLM navigates the AST graph via get_neighbors and records code mutations
    (add_node / modify_node / delete_node) before calling submit.
    Each episode writes a timestamped log file to the project root.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._current_graph: Optional[nx.DiGraph] = None
        self._state: Optional[ASTState] = None

    # -------------------------------------------------------------------------
    # OpenENV interface
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ASTObservation:
        query_obj, pkl_path = self._pick_random_query(seed)

        with open(pkl_path, "rb") as fh:
            self._current_graph = pickle.load(fh)

        context_nodes, context_edges = self._seed_context(query_obj)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = os.path.join(_PROJECT_ROOT, f"logs-{timestamp}.txt")
        eid = episode_id or str(uuid.uuid4())

        self._state = ASTState(
            episode_id=eid,
            step_count=0,
            query_obj=query_obj,
            graph_pkl_path=pkl_path,
            base_dir=_SAMPLE_DIR,
            context_nodes=context_nodes,
            context_edges=context_edges,
            not_found_nodes=[],
            mutations=[],
            is_terminal=False,
            log_path=log_path,
        )

        self._log("=" * 70)
        self._log(f"EPISODE START  id={eid}")
        self._log(f"Query      : {self._query_text()}")
        self._log(f"Central    : {query_obj.get('central_node', '')}")
        self._log(f"Graph      : {os.path.basename(pkl_path)}")
        self._log(f"Context    : {len(context_nodes)} nodes, {len(context_edges)} edges")
        self._log("=" * 70)

        return self._build_obs(result=None)

    def step(
        self,
        action: ASTAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ASTObservation:
        if self._state is None:
            return ASTObservation(done=True, metadata={"error": "call reset() first"})

        state = self._state
        result = self._dispatch(action)
        state.step_count += 1

        # Log step details (omit code body from params line to keep log readable)
        safe_params = {k: v for k, v in action.parameters.items() if k != "code"}
        self._log(f"\n[Step {state.step_count}] {action.action_type}")
        self._log(f"  params   : {json.dumps(safe_params)}")
        self._log(f"  result   : {result.message}")
        if action.action_type in ("add_node", "modify_node", "delete_node"):
            self._log(f"  reasoning: {action.parameters.get('reasoning', '')}")
        if action.action_type in ("add_node", "modify_node") and action.parameters.get("code"):
            self._log(f"  code     :\n{action.parameters['code']}")

        if action.action_type == "submit" or state.step_count >= MAX_STEPS:
            state.is_terminal = True
            self._log("\n" + "=" * 70)
            self._log("EPISODE END")
            self._log(f"Steps      : {state.step_count}")
            self._log(f"Mutations  : {len(state.mutations)}")
            for i, m in enumerate(state.mutations, 1):
                self._log(f"  [{i}] {m.get('type'):<8} {m.get('node_id')}")
                self._log(f"       {m.get('reasoning', '')}")
            self._log("=" * 70)

        return self._build_obs(result)

    @property
    def state(self) -> ASTState:
        return self._state or ASTState()

    # -------------------------------------------------------------------------
    # Action dispatch
    # -------------------------------------------------------------------------

    def _dispatch(self, action: ASTAction) -> ASTActionResult:
        state = self._state
        at = action.action_type
        p = action.parameters

        if at == "get_neighbors":
            node_id = p.get("node_id", "")
            new_nodes, new_edges = _get_neighbors(
                self._current_graph, node_id, state.base_dir
            )
            if not new_nodes:
                if node_id not in state.not_found_nodes:
                    state.not_found_nodes.append(node_id)
                return ASTActionResult(
                    success=False,
                    action_type=at,
                    message=f"Node not found in graph: {node_id!r}",
                )
            state.context_nodes.update(new_nodes)
            existing_keys = {(e["source"], e["target"]) for e in state.context_edges}
            for e in new_edges:
                key = (e["source"], e["target"])
                if key not in existing_keys:
                    state.context_edges.append(e)
                    existing_keys.add(key)
            return ASTActionResult(
                success=True,
                action_type=at,
                message=f"Added {len(new_nodes)} node(s) to context",
                added_node_ids=list(new_nodes.keys()),
            )

        if at == "add_node":
            mutation = {
                k: v
                for k, v in {
                    "type": "ADD",
                    "node_id": p.get("node_id", ""),
                    "parent_id": p.get("parent_id"),
                    "child_ids": p.get("child_ids"),
                    "code": p.get("code"),
                    "reasoning": p.get("reasoning", ""),
                }.items()
                if v is not None
            }
            state.mutations.append(mutation)
            return ASTActionResult(
                success=True,
                action_type=at,
                message=f"ADD mutation recorded for {p.get('node_id')}",
            )

        if at == "modify_node":
            mutation = {
                k: v
                for k, v in {
                    "type": "MODIFY",
                    "node_id": p.get("node_id", ""),
                    "code": p.get("code"),
                    "reasoning": p.get("reasoning", ""),
                }.items()
                if v is not None
            }
            state.mutations.append(mutation)
            return ASTActionResult(
                success=True,
                action_type=at,
                message=f"MODIFY mutation recorded for {p.get('node_id')}",
            )

        if at == "delete_node":
            mutation = {
                "type": "DELETE",
                "node_id": p.get("node_id", ""),
                "reasoning": p.get("reasoning", ""),
            }
            state.mutations.append(mutation)
            return ASTActionResult(
                success=True,
                action_type=at,
                message=f"DELETE mutation recorded for {p.get('node_id')}",
            )

        if at == "submit":
            return ASTActionResult(
                success=True,
                action_type=at,
                message=f"Episode submitted with {len(state.mutations)} mutation(s)",
            )

        return ASTActionResult(
            success=False, action_type=at, message=f"Unknown action: {at!r}"
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _pick_random_query(self, seed: Optional[int]) -> Tuple[Dict[str, Any], str]:
        mapping_path = os.path.join(_SAMPLE_DIR, "project_mapping.json")
        with open(mapping_path) as fh:
            mapping = json.load(fh)

        entries: List[Tuple[Dict[str, Any], str]] = []
        for _item_name, info in mapping.items():
            q_rel = info.get("queries")
            ast_rel = info.get("ast_graph")
            if not q_rel or not ast_rel:
                continue
            q_path = os.path.join(_SAMPLE_DIR, q_rel)
            pkl_path = os.path.join(_SAMPLE_DIR, ast_rel)
            if not os.path.exists(q_path) or not os.path.exists(pkl_path):
                continue
            with open(q_path) as fh:
                queries = json.load(fh)
            if not isinstance(queries, list):
                queries = [queries]
            for q in queries:
                entries.append((q, pkl_path))

        if not entries:
            raise RuntimeError(
                f"No query entries found. Checked mapping at: {mapping_path}"
            )

        return random.Random(seed).choice(entries)

    def _seed_context(
        self, query_obj: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        subgraph = (
            query_obj.get("subgraph")
            or query_obj.get("context_subgraph")
            or {"nodes": {}, "edges": []}
        )
        context_nodes, context_edges = _normalize_subgraph(subgraph)

        for nid in list(context_nodes.keys()):
            src = _get_node_source(nid, _SAMPLE_DIR)
            if src:
                context_nodes[nid]["source"] = src

        central_node = query_obj.get("central_node")
        if central_node and self._current_graph is not None:
            seed_nodes, seed_edges = _get_neighbors(
                self._current_graph, central_node, _SAMPLE_DIR
            )
            context_nodes.update(seed_nodes)
            existing = {(e["source"], e["target"]) for e in context_edges}
            for ne in seed_edges:
                if (ne["source"], ne["target"]) not in existing:
                    context_edges.append(ne)

        return context_nodes, context_edges

    def _query_text(self) -> str:
        if not self._state:
            return ""
        q = self._state.query_obj
        return q.get("task") or q.get("prompt") or ""

    def _log(self, msg: str) -> None:
        if self._state and self._state.log_path:
            with open(self._state.log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    def _build_obs(self, result: Optional[ASTActionResult]) -> ASTObservation:
        state = self._state
        available = (
            []
            if state.is_terminal
            else list(ALL_AST_ACTION_TYPES)
        )
        return ASTObservation(
            done=state.is_terminal,
            reward=float(len(state.mutations)) if state.is_terminal else None,
            metadata={"step_count": state.step_count, "log_path": state.log_path},
            episode_id=state.episode_id,
            timestep=state.step_count,
            query=self._query_text(),
            central_node=state.query_obj.get("central_node", ""),
            context_nodes=state.context_nodes,
            context_edges=state.context_edges,
            not_found_nodes=state.not_found_nodes,
            mutations_so_far=state.mutations,
            last_action_result=result,
            available_actions=available,
        )

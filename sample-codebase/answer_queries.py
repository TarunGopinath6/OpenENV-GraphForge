import os
from dotenv import load_dotenv
import ast
import json
import pickle
import networkx as nx
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Optional, List

OPENAI_API_KEY = "PASTE_YOUR_KEY_HERE"

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class Mutation(BaseModel):
    type: Literal["ADD", "MODIFY", "DELETE"]
    node_id: str
    parent_id: Optional[str] = None
    child_ids: Optional[List[str]] = None
    code: Optional[str] = None  # Not required for DELETE
    reasoning: str


class LLMResponse(BaseModel):
    status: Literal["NEED_CONTEXT", "FINAL_ANSWER"]
    requested_node_ids: Optional[List[str]] = None
    mutations: Optional[List[Mutation]] = None
    reasoning: Optional[str] = None


def load_mapping():
    mapping_path = "project_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            return json.load(f)
    return {}


def get_node_source(node_id: str, base_dir: str) -> Optional[str]:
    """
    Given a node_id like './codebases/file.py::symbol', read the source file
    and return the exact source lines for that function/class using Python's ast.
    Returns None for module-level nodes or if the symbol can't be found.
    """
    if "::" not in node_id:
        return None

    file_part, symbol = node_id.rsplit("::", 1)
    file_path = os.path.join(base_dir, file_part)

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    except Exception:
        pass
    return None


def get_neighbors(G, node_id: str, base_dir: str):
    """Returns nodes (with source code) and edges adjacent to node_id."""
    if node_id not in G:
        return {}, []

    neighbors = list(G.successors(node_id)) + list(G.predecessors(node_id))

    new_nodes = {}
    new_edges = []

    for n in neighbors + [node_id]:
        node_data = {"type": G.nodes[n].get("type", "unknown")}
        source = get_node_source(n, base_dir)
        if source:
            node_data["source"] = source
        new_nodes[n] = node_data

    for u, v in G.edges(neighbors + [node_id]):
        if u in new_nodes and v in new_nodes:
            new_edges.append({"source": u, "target": v, "attributes": G.edges[u, v]})

    return new_nodes, new_edges


def normalize_subgraph(subgraph: dict):
    """
    Normalise the two subgraph formats found in query JSON files:
      nodes  — dict {id: metadata}  OR  list of id strings
      edges  — list of {source, target, ...} dicts  OR  list of [src, tgt] lists
    Returns (nodes_dict, edges_list) in the canonical dict form.
    """
    nodes_raw = subgraph.get("nodes", {})
    if isinstance(nodes_raw, list):
        nodes = {nid: {} for nid in nodes_raw}
    else:
        nodes = dict(nodes_raw)

    edges_raw = subgraph.get("edges", [])
    edges = []
    for e in edges_raw:
        if isinstance(e, (list, tuple)):
            edges.append({"source": e[0], "target": e[1], "attributes": {}})
        else:
            edges.append(dict(e))

    return nodes, edges


def process_query(query_obj, G, base_dir: str):
    """Iterative loop for LLM reasoning."""
    subgraph = query_obj.get("subgraph") or query_obj.get("context_subgraph") or {"nodes": {}, "edges": []}
    context_nodes, context_edges = normalize_subgraph(subgraph)
    not_found_nodes: set = set()

    # Enrich any nodes already in context with their source code
    for nid in list(context_nodes.keys()):
        source = get_node_source(nid, base_dir)
        if source:
            context_nodes[nid]["source"] = source

    # Bootstrap: seed neighbors of the central node so the LLM starts with real data
    central_node = query_obj.get("central_node")
    if central_node:
        seed_nodes, seed_edges = get_neighbors(G, central_node, base_dir)
        context_nodes.update(seed_nodes)
        existing_keys = set((e["source"], e["target"]) for e in context_edges)
        for ne in seed_edges:
            if (ne["source"], ne["target"]) not in existing_keys:
                context_edges.append(ne)

    iteration = 0

    while True:
        iteration += 1

        not_found_hint = ""
        if not_found_nodes:
            not_found_hint = (
                f"\nNODE IDs NOT FOUND IN GRAPH — do not request these again: "
                f"{json.dumps(list(not_found_nodes))}\n"
            )

        available_hint = ""
        if not context_nodes:
            sample = list(G.nodes())[:40]
            available_hint = (
                f"\nNO CONTEXT LOADED YET. Sample of real node IDs in the graph "
                f"(request one or more to get started):\n{json.dumps(sample, indent=2)}\n"
            )

        prompt_text = f"""
        You are an AST-driven code evolution agent.

        TASK: {query_obj.get('task') or query_obj.get('prompt')}
        CENTRAL NODE: {query_obj.get('central_node')}

        CURRENT AST CONTEXT:
        Nodes: {json.dumps(context_nodes, indent=2)}
        Edges: {json.dumps(context_edges, indent=2)}
        {not_found_hint}{available_hint}
        INSTRUCTIONS:
        1. Analyze if you have enough information to fulfill the task.
        2. If you need more context, set "status" to "NEED_CONTEXT" and list real node IDs from the context above in "requested_node_ids". Only request IDs that already appear in the current context — do not invent or guess node IDs.
        3. Once you have enough context, set "status" to "FINAL_ANSWER". Provide ALL required mutations together in the "mutations" list — you may mix ADD, MODIFY, and DELETE operations in a single response.

        MUTATION STRUCTURE (each entry in the "mutations" list):
        - type: "ADD" | "MODIFY" | "DELETE"
        - node_id: Target identifier (new ID for ADD, existing ID for MODIFY/DELETE)
        - parent_id: (ADD only, optional) ID of the parent node
        - child_ids: (ADD only, optional) Existing children to re-parent under the new node
        - code: Python code snippet — required for ADD and MODIFY, omit for DELETE
        - reasoning: Technical justification for this specific mutation

        Order mutations so that ADD operations come before any MODIFY or DELETE that depends on them.

        Respond ONLY with a JSON object.
        """

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a technical assistant for AST manipulation. You output structured JSON."},
                    {"role": "user", "content": prompt_text}
                ],
                response_format=LLMResponse
            )

            res: LLMResponse = response.choices[0].message.parsed

            if res.status == "NEED_CONTEXT":
                print(f"  [Iteration {iteration}] LLM requested: {res.requested_node_ids}")
                existing_edge_keys = set((e["source"], e["target"]) for e in context_edges)
                for node_id in (res.requested_node_ids or []):
                    new_nodes, new_edges = get_neighbors(G, node_id, base_dir)
                    if not new_nodes:
                        not_found_nodes.add(node_id)
                        print(f"    [!] Node not found in graph: {node_id}")
                    else:
                        print(f"    [+] Added {len(new_nodes)} node(s): {list(new_nodes.keys())}")
                        for nid, ndata in new_nodes.items():
                            has_source = "source" in ndata
                            print(f"        {nid!r} | type={ndata.get('type')} | has_source={has_source}")
                        context_nodes.update(new_nodes)
                        for ne in new_edges:
                            if (ne["source"], ne["target"]) not in existing_edge_keys:
                                context_edges.append(ne)
                                existing_edge_keys.add((ne["source"], ne["target"]))

            elif res.status == "FINAL_ANSWER":
                print(f"  [Iteration {iteration}] LLM provided final answer.")
                mutations = [m.model_dump(exclude_none=True) for m in (res.mutations or [])]
                return mutations, res.reasoning

        except Exception as e:
            print(f"  Error in LLM loop: {e}")
            return None, str(e)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mapping = load_mapping()

    query_files_to_items = {}
    for item, info in mapping.items():
        q_path = info.get("queries")
        if q_path:
            query_files_to_items[q_path] = item

    for q_path, item_name in query_files_to_items.items():
        full_q_path = os.path.join(base_dir, q_path)
        if not os.path.exists(full_q_path):
            continue

        print(f"\n--- Processing Query File: {q_path} (Item: {item_name}) ---")

        ast_rel_path = mapping[item_name].get("ast_graph")
        if not ast_rel_path:
            continue
        ast_full_path = os.path.join(base_dir, ast_rel_path)

        with open(ast_full_path, "rb") as f:
            G = pickle.load(f)

        with open(full_q_path, "r") as f:
            queries = json.load(f)

        if not isinstance(queries, list):
            queries = [queries]

        updated = False
        for q in queries:
            if "llm_result" in q:
                continue

            print(f"Processing Query: {q.get('task') or q.get('prompt')}")
            mutations, reasoning = process_query(q, G, base_dir)

            q["llm_result"] = {
                "total_actions": len(mutations) if mutations else 0,
                "actions": mutations or [],
                "reasoning": reasoning
            }
            updated = True

        if updated:
            with open(full_q_path, "w") as f:
                json.dump(queries, f, indent=4)
            print(f"Updated {q_path} with results.")


if __name__ == "__main__":
    main()

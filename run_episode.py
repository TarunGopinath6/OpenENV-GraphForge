"""Run one AST-query episode with Claude as the agent.

Usage:
    python run_episode.py
    python run_episode.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os

import anthropic
from dotenv import load_dotenv

from models import ASTAction, ASTObservation
from server.ast_environment import ASTEnvironment

load_dotenv()

MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tool definitions — map 1-to-1 to ASTAction.action_type values
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_neighbors",
        "description": (
            "Fetch the neighboring nodes and edges of a node from the AST graph. "
            "Use this to expand context before making mutations. "
            "Only request node IDs that already appear in your current context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "A node ID from your current context (e.g. './codebases/foo.py::bar').",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "add_node",
        "description": "Record an ADD mutation: introduce a new function or class to the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "New node ID (e.g. './codebases/foo.py::new_func').",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent module or class node ID (optional).",
                },
                "child_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing child nodes to re-parent under the new node (optional).",
                },
                "code": {
                    "type": "string",
                    "description": "Complete Python source code for the new node.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Technical justification for adding this node.",
                },
            },
            "required": ["node_id", "code", "reasoning"],
        },
    },
    {
        "name": "modify_node",
        "description": "Record a MODIFY mutation: update the source of an existing node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Existing node ID to modify.",
                },
                "code": {
                    "type": "string",
                    "description": "Complete updated Python source (full function/class, not a diff).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Technical justification for this change.",
                },
            },
            "required": ["node_id", "code", "reasoning"],
        },
    },
    {
        "name": "delete_node",
        "description": "Record a DELETE mutation: remove an existing node from the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Existing node ID to delete.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Technical justification for deletion.",
                },
            },
            "required": ["node_id", "reasoning"],
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit your final answer. Call this once all required mutations are recorded. "
            "You may mix ADD, MODIFY, and DELETE operations before submitting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """\
You are an AST-driven code evolution agent working on a real Python codebase.

Strategy:
1. Read the task and the initial context (node source code + edges) carefully.
2. Call get_neighbors on nodes already in your context to gather more detail if needed.
   Do NOT invent node IDs — only request IDs that appear in your current context.
3. Record all required mutations using add_node, modify_node, or delete_node.
   - add_node: provide complete, runnable Python source.
   - modify_node: provide the full updated source, not a diff.
   - Order ADD mutations before any MODIFY/DELETE that depends on them.
4. Call submit when done.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_user_message(obs: ASTObservation) -> str:
    not_found = (
        f"\nNOT FOUND — do not request again: {json.dumps(obs.not_found_nodes)}\n"
        if obs.not_found_nodes
        else ""
    )
    recorded = (
        f"\nMUTATIONS RECORDED ({len(obs.mutations_so_far)}):\n"
        + json.dumps(
            [{"type": m.get("type"), "node_id": m.get("node_id")} for m in obs.mutations_so_far],
            indent=2,
        )
        + "\n"
        if obs.mutations_so_far
        else ""
    )
    return (
        f"TASK: {obs.query}\n"
        f"CENTRAL NODE: {obs.central_node}\n\n"
        f"CONTEXT NODES ({len(obs.context_nodes)}):\n"
        f"{json.dumps(list(obs.context_nodes.keys()), indent=2)}\n\n"
        f"NODE DETAILS:\n{json.dumps(obs.context_nodes, indent=2)}\n\n"
        f"EDGES:\n{json.dumps(obs.context_edges, indent=2)}"
        f"{not_found}{recorded}"
        f"\nAnalyze the task and proceed."
    )


def _tool_result(tu_id: str, obs: ASTObservation, prev_count: int) -> dict:
    r = obs.last_action_result
    msg = r.message if r else "Done."
    if r and r.added_node_ids:
        new_data = {nid: obs.context_nodes.get(nid, {}) for nid in r.added_node_ids}
        msg += f"\nNew node details:\n{json.dumps(new_data, indent=2)}"
        msg += (
            f"\nContext now has {len(obs.context_nodes)} nodes "
            f"(+{len(obs.context_nodes) - prev_count})."
        )
    return {"type": "tool_result", "tool_use_id": tu_id, "content": msg}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run(seed: int | None = None) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    env = ASTEnvironment()
    obs = env.reset(seed=seed)

    print(f"\n{'='*60}")
    print(f"EPISODE START")
    print(f"Query   : {obs.query}")
    print(f"Central : {obs.central_node}")
    print(f"Log     : {obs.metadata.get('log_path', '')}")
    print(f"{'='*60}\n")

    messages: list = [{"role": "user", "content": _obs_to_user_message(obs)}]

    while not obs.done:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            print("[!] No tool call — forcing submit.")
            obs = env.step(ASTAction(action_type="submit", parameters={}))
            break

        tool_results = []
        for tu in tool_uses:
            prev_count = len(obs.context_nodes)
            action = ASTAction(action_type=tu.name, parameters=tu.input)  # type: ignore[arg-type]
            obs = env.step(action)
            result_msg = obs.last_action_result.message if obs.last_action_result else ""
            print(f"[Step {obs.timestep:>3}] {tu.name:<16} {result_msg}")
            tool_results.append(_tool_result(tu.id, obs, prev_count))
            if obs.done:
                break

        if not obs.done:
            summary = (
                f"\n\nContext: {len(obs.context_nodes)} nodes. "
                f"Not found: {obs.not_found_nodes}. "
                f"Mutations: {len(obs.mutations_so_far)}."
            )
            tool_results[-1]["content"] += summary
            messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*60}")
    print(f"DONE — {obs.timestep} step(s)")
    print(f"Mutations ({len(obs.mutations_so_far)}):")
    for i, m in enumerate(obs.mutations_so_far, 1):
        print(f"  [{i}] {m.get('type'):<8} {m.get('node_id')}")
    print(f"Log : {obs.metadata.get('log_path', '')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one AST-query episode with Claude.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for query selection.")
    args = parser.parse_args()
    run(seed=args.seed)

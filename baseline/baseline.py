"""Baseline policies for GraphForge environment."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from models import ALL_ACTION_TYPES, GraphAction, GraphObservation


class RandomBaseline:
    """Selects random actions with structurally valid parameters."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, obs: GraphObservation) -> GraphAction:
        available = obs.available_actions or ALL_ACTION_TYPES
        action_type = self._rng.choice(available)
        params = self._random_params(action_type, obs)
        return GraphAction(action_type=action_type, parameters=params)  # type: ignore[arg-type]

    def _random_params(self, action_type: str, obs: GraphObservation) -> Dict[str, Any]:
        modules = [m.name for m in obs.graph_state.modules]
        nodes = [f"{n.module}.{n.name}" for n in obs.graph_state.nodes]
        edges = [(e.caller, e.callee) for e in obs.graph_state.edges]

        if action_type == "add_module":
            return {"name": f"mod_{self._rng.randint(0, 999)}", "responsibility": self._rng.choice(["transform", "validation", "io", "orchestration"])}
        if action_type == "remove_module":
            return {"name": self._rng.choice(modules) if modules else "unknown"}
        if action_type == "add_node":
            mod = self._rng.choice(modules) if modules else "unknown"
            return {"name": f"func_{self._rng.randint(0, 999)}", "module": mod, "signature": "(x: int) -> int", "purity": "unknown", "error_policy": "raise"}
        if action_type == "remove_node":
            chosen = self._rng.choice(nodes).split(".") if nodes else ["unknown", "unknown"]
            return {"name": chosen[-1], "module": chosen[0] if len(chosen) > 1 else "unknown"}
        if action_type == "set_node_module":
            node = self._rng.choice(nodes).split(".") if nodes else ["unknown", "unknown"]
            new_mod = self._rng.choice(modules) if modules else "unknown"
            return {"name": node[-1], "current_module": node[0] if len(node) > 1 else "unknown", "new_module": new_mod}
        if action_type == "attach_body":
            node = self._rng.choice(nodes).split(".") if nodes else ["unknown", "unknown"]
            return {"name": node[-1], "module": node[0] if len(node) > 1 else "unknown", "template": "identity", "args": {"param": "x"}}
        if action_type == "add_edge":
            if len(nodes) >= 2:
                caller, callee = self._rng.sample(nodes, 2)
            else:
                caller = callee = nodes[0] if nodes else "unknown.unknown"
            return {"caller": caller, "callee": callee, "arg_mapping": []}
        if action_type == "remove_edge":
            if edges:
                caller, callee = self._rng.choice(edges)
            else:
                caller = callee = "unknown.unknown"
            return {"caller": caller, "callee": callee}
        if action_type == "query_spec":
            return {}
        if action_type == "query_subgraph":
            if modules:
                return {"scope": f"module:{self._rng.choice(modules)}"}
            return {"scope": "module:unknown"}
        if action_type == "query_types":
            return {"scope": "all"}
        if action_type in ("materialize_and_validate", "run_behavioral_tests", "submit"):
            return {}
        return {}


class GreedyConstraintBaseline:
    """Tries to satisfy the cheapest unsatisfied visible constraint each turn.

    Falls back to RandomBaseline when no obvious greedy action is available.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._random = RandomBaseline(seed=seed)

    def select_action(self, obs: GraphObservation) -> GraphAction:
        # Find an unsatisfied constraint and suggest the obvious fix
        from engine.constraint_checker import ConstraintChecker
        cc = ConstraintChecker()

        for constraint in obs.visible_constraints:
            satisfied = cc.check_one(obs.graph_state, constraint)
            if not satisfied:
                action = self._action_for_constraint(constraint, obs)
                if action is not None:
                    return action

        # No greedy opportunity — fall back
        return self._random.select_action(obs)

    def _action_for_constraint(self, constraint, obs: GraphObservation) -> Optional[GraphAction]:
        kind = constraint.kind
        target = constraint.target

        if kind == "node_exists" and "." in target:
            module, name = target.split(".", 1)
            # Ensure the module exists first
            if not any(m.name == module for m in obs.graph_state.modules):
                return GraphAction(action_type="add_module", parameters={"name": module, "responsibility": "transform"})
            # Then add the node
            if not any(n.module == module and n.name == name for n in obs.graph_state.nodes):
                return GraphAction(action_type="add_node", parameters={
                    "name": name, "module": module,
                    "signature": "(x: int) -> int", "purity": "unknown", "error_policy": "raise",
                })

        if kind == "edge_exists" and "::" in target:
            caller, callee = target.split("::", 1)
            return GraphAction(action_type="add_edge", parameters={"caller": caller.strip(), "callee": callee.strip(), "arg_mapping": []})

        if kind == "module_responsibility":
            return GraphAction(action_type="add_module", parameters={"name": target, "responsibility": str(constraint.value)})

        return None

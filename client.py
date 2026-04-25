"""GraphForge OpenENV client."""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from models import GraphAction, GraphForgeState, GraphObservation


class GraphForgeClient(EnvClient[GraphAction, GraphObservation, GraphForgeState]):
    """WebSocket client for the GraphForge environment server."""

    def _step_payload(self, action: GraphAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GraphObservation]:
        obs = GraphObservation.model_validate(payload)
        return StepResult(
            observation=obs,
            reward=obs.reward if isinstance(obs.reward, (int, float)) else None,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GraphForgeState:
        return GraphForgeState.model_validate(payload)

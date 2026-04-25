"""GraphForge OpenENV-compliant environment."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Environment

from models import (
    ALL_ACTION_TYPES,
    ActionResult,
    ConstraintSummary,
    GraphAction,
    GraphForgeState,
    GraphObservation,
    GraphState,
)
from engine.action_dispatcher import ActionDispatcher
from engine.behavioral_test_runner import BehavioralTestRunner
from engine.body_template_library import BodyTemplateLibrary
from engine.constraint_checker import ConstraintChecker
from engine.materializer import Materializer
from engine.reward_engine import RewardEngine
from engine.token_counter import TokenCounter
from engine.type_engine import TypeEngine
from engine.validator import Validator
from tasks.task_bank import TaskBank


class GraphForgeEnvironment(Environment[GraphAction, GraphObservation, GraphForgeState]):
    """Full OpenENV-compliant environment for graph-driven code generation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        # Instantiate all engine components
        te = TypeEngine()
        btl = BodyTemplateLibrary()
        mat = Materializer()
        val = Validator()
        tr = BehavioralTestRunner()
        cc = ConstraintChecker()
        re = RewardEngine()
        tc = TokenCounter()

        self._dispatcher = ActionDispatcher(te, btl, mat, val, tr, cc, re, tc)
        self._re = re
        self._tc = tc
        self._task_bank = TaskBank()
        self._state: Optional[GraphForgeState] = None

    # -------------------------------------------------------------------------
    # OpenENV interface
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        tier: int = 1,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GraphObservation:
        if task_id:
            task_spec = self._task_bank.get_task(task_id)
        else:
            task_spec = self._task_bank.sample(tier=tier, seed=seed)

        eid = episode_id or str(uuid.uuid4())
        self._state = GraphForgeState(
            episode_id=eid,
            step_count=0,
            graph=GraphState(),
            task_spec=task_spec,
            turn_cap=task_spec.turn_cap,
            token_budget=task_spec.token_budget,
            tokens_used=0,
            cumulative_reward=0.0,
            is_terminal=False,
            materialization_cache=None,
            last_mypy_output=None,
            last_test_results=None,
            action_history=[],
        )
        return self._build_observation(action_result=None)

    def step(
        self,
        action: GraphAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GraphObservation:
        if self._state is None:
            return GraphObservation(
                done=True,
                reward=-2.0,
                metadata={"error": "call reset() first"},
            )

        state = self._state

        # Token cost for this action
        action_tokens = self._tc.count_action(action)
        state.tokens_used += action_tokens

        # Validate action type
        if action.action_type not in ALL_ACTION_TYPES:
            penalty = self._re.malformed_action_penalty()
            state.cumulative_reward += penalty
            obs = self._build_observation(
                ActionResult(success=False, error_kind="unknown_action",
                             error_msg=f"Unknown action: {action.action_type!r}", penalty=penalty)
            )
            return obs

        # Check repeat action
        action_key = (action.action_type, json.dumps(action.parameters, sort_keys=True, default=str))
        is_repeat = action_key in state.action_history
        if not is_repeat:
            state.action_history.append(action_key)

        # Dispatch
        state, result = self._dispatcher.dispatch(state, action, is_repeat=is_repeat)
        self._state = state

        # Per-turn reward
        per_turn = self._re.per_turn_reward(result, action_tokens, is_repeat)
        state.cumulative_reward += per_turn

        # Increment step count
        state.step_count += 1

        # Check turn cap
        if state.step_count >= state.turn_cap and not state.is_terminal:
            state.is_terminal = True

        # Check token budget
        if state.tokens_used >= state.token_budget and not state.is_terminal:
            state.is_terminal = True

        obs = self._build_observation(result)
        # Count response tokens
        obs_tokens = self._tc.count_observation(obs)
        state.tokens_used += obs_tokens

        return obs

    @property
    def state(self) -> GraphForgeState:
        if self._state is None:
            return GraphForgeState()
        return self._state

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_observation(self, action_result: Optional[ActionResult]) -> GraphObservation:
        if self._state is None:
            return GraphObservation(done=True)

        state = self._state
        task = state.task_spec

        # Constraint summary (visible only)
        from engine.constraint_checker import ConstraintChecker
        cc = ConstraintChecker()
        visible = [c for c in (task.constraints if task else []) if not c.hidden]
        satisfied, total = cc.count_satisfied(state.graph, visible)
        summary = ConstraintSummary(
            total=len(task.constraints) if task else 0,
            visible=len(visible),
            hidden=len(task.constraints) - len(visible) if task else 0,
            satisfied_visible=satisfied,
        )

        # Available actions depend on state
        available = list(ALL_ACTION_TYPES)
        if state.materialization_cache is None:
            try:
                available.remove("run_behavioral_tests")
            except ValueError:
                pass

        return GraphObservation(
            done=state.is_terminal,
            reward=state.cumulative_reward if state.is_terminal else None,
            metadata={
                "tokens_used": state.tokens_used,
                "step_count": state.step_count,
            },
            episode_id=state.episode_id or "",
            timestep=state.step_count,
            turn_budget_remaining=max(0, state.turn_cap - state.step_count),
            token_budget_remaining=max(0, state.token_budget - state.tokens_used),
            graph_state=state.graph,
            last_action_result=action_result,
            constraint_summary=summary,
            visible_constraints=visible,
            available_actions=available,
            task_description=task.description if task else "",
        )

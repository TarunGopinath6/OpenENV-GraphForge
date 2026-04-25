"""Reward engine: computes per-turn and terminal rewards."""

from __future__ import annotations

from typing import List, Optional

from models import ActionResult, MaterializeResult, ConstraintSpec, GraphState, TestResult


class RewardEngine:
    """Implements the GraphForge reward structure."""

    # Per-turn penalties
    MUTATION_FAIL: float = -2.0
    MALFORMED_ACTION: float = -2.0
    REPEAT_ACTION: float = -1.0
    PER_TURN: float = -0.1
    TOKEN_ALPHA: float = 0.001

    # Terminal bonuses / penalties
    STRUCTURAL_PER: float = 1.0
    BEHAVIORAL_PER: float = 3.0
    ALL_STRUCTURAL_BONUS: float = 5.0
    ALL_BEHAVIORAL_BONUS: float = 5.0
    TYPE_CHECK_BONUS: float = 3.0
    MATERIALIZE_FAIL_PENALTY: float = -8.0
    TOKEN_EFFICIENCY_BONUS: float = 5.0

    def per_turn_reward(
        self,
        action_result: ActionResult,
        tokens_this_turn: int = 0,
        is_repeat: bool = False,
    ) -> float:
        r = self.PER_TURN
        r -= self.TOKEN_ALPHA * max(0, tokens_this_turn)
        if is_repeat:
            r += self.REPEAT_ACTION
        if not action_result.success:
            r += self.MUTATION_FAIL
        return r

    def malformed_action_penalty(self) -> float:
        return self.MALFORMED_ACTION + self.PER_TURN

    def terminal_reward(
        self,
        graph: GraphState,
        constraints: List[ConstraintSpec],
        test_results: List[TestResult],
        materialize_result: MaterializeResult,
        mypy_ok: bool,
        tokens_used: int,
        token_budget: int,
    ) -> float:
        if not materialize_result.success:
            return self.MATERIALIZE_FAIL_PENALTY

        r = 0.0

        # Structural constraints (visible only — hidden evaluated at submit)
        visible = [c for c in constraints if not c.hidden]
        from engine.constraint_checker import ConstraintChecker
        checker = ConstraintChecker()
        satisfied_map = checker.check_all(graph, visible)
        n_satisfied = sum(satisfied_map.values())
        n_total = len(visible)

        r += n_satisfied * self.STRUCTURAL_PER
        if n_total > 0 and n_satisfied == n_total:
            r += self.ALL_STRUCTURAL_BONUS

        # Behavioral tests
        n_pass = sum(1 for tr in test_results if tr.passed)
        n_tests = len(test_results)
        r += n_pass * self.BEHAVIORAL_PER
        if n_tests > 0 and n_pass == n_tests:
            r += self.ALL_BEHAVIORAL_BONUS

        # Type check
        if mypy_ok:
            r += self.TYPE_CHECK_BONUS

        # Token efficiency bonus (only if all structural + behavioral pass)
        if n_tests + n_total > 0 and n_satisfied == n_total and n_pass == n_tests:
            r += self.compute_token_efficiency_bonus(tokens_used, token_budget)

        return r

    def compute_token_efficiency_bonus(self, tokens_used: int, token_budget: int) -> float:
        if token_budget <= 0:
            return 0.0
        fraction_remaining = max(0.0, (token_budget - tokens_used) / token_budget)
        return self.TOKEN_EFFICIENCY_BONUS * fraction_remaining

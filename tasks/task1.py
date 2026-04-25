"""Basic task — delegates to TaskBank for Tier 1 task sampling."""

from typing import Dict

from tasks.task_bank import TaskBank


class BasicTask:
    """Tier 1 task wrapper delegating to TaskBank."""

    def __init__(self) -> None:
        self.name = "basic_task"
        self.difficulty = "easy"
        self._bank = TaskBank()

    def sample(self) -> Dict:
        return self._bank.sample(tier=1).model_dump()

    def evaluate(self, trajectory: Dict) -> float:
        return 0.0

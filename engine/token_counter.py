"""Token counter: server-side token accounting using the Qwen2.5 tokenizer."""

from __future__ import annotations

import json
from typing import Optional

from models import GraphAction, GraphObservation


class TokenCounter:
    """Counts tokens in strings and serialized observations/actions.

    Uses the Qwen2.5-0.5B-Instruct tokenizer when available; falls back to
    a character-length approximation (1 token per 4 chars) otherwise.
    """

    _MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._tokenizer = None
        name = model_name or self._MODEL_NAME
        try:
            from transformers import AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception:
            pass  # fall back to approximation

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)

    def count_observation(self, obs: GraphObservation) -> int:
        return self.count(obs.model_dump_json())

    def count_action(self, action: GraphAction) -> int:
        return self.count(action.model_dump_json())

    def count_dict(self, d: dict) -> int:
        return self.count(json.dumps(d, default=str))

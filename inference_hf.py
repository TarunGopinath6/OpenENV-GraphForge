#!/usr/bin/env python3
"""
Inference Script — GraphForge AST Mutation Agent (HF Space edition)
=====================================================================
Identical logic to inference.py but connects to the remote OpenENV server
running on the Hugging Face Space instead of instantiating ASTEnvironment
locally.

MANDATORY environment variables:
    HF_SPACE_URL      URL of the running HF Space (defaults below).
    API_BASE_URL      LLM API endpoint (OpenAI-compatible).
    MODEL_NAME        Model identifier for inference.
    HF_TOKEN          Hugging Face / API key.

Defaults:
    HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://tarungopinath-openenv-graphforge.hf.space")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
"""

import ast as _ast
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.env_client import StepResult

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from models import ASTAction, ASTObservation, ASTState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL", "https://tarungopinath-openenv-graphforge.hf.space"
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK = "graphforge-ast"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STEPS_PER_EPISODE = 20

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AST-driven code mutation agent. You analyze real Python codebases and propose
    targeted mutations to satisfy a given task.

    Each turn you will see:
      - TASK: the refactoring or enhancement task
      - CENTRAL NODE: the primary node to modify
      - CONTEXT NODES: node IDs currently available in your context
      - NODE DETAILS: source code and metadata for those nodes

    Respond with EXACTLY ONE JSON object on a single line.

    Available actions:

    Explore a node's neighbors (to get more source context):
      {"action": "get_neighbors", "node_id": "<node_id_from_context>"}

    Add a new function or class:
      {"action": "add_node", "node_id": "<new_node_id>", "code": "<full_python_source>", "reasoning": "<why>"}

    Modify an existing node:
      {"action": "modify_node", "node_id": "<existing_node_id>", "code": "<full_updated_source>", "reasoning": "<why>"}

    Delete an existing node:
      {"action": "delete_node", "node_id": "<existing_node_id>", "reasoning": "<why>"}

    Submit all recorded mutations (call when done):
      {"action": "submit"}

    Rules:
      - Only use node IDs that appear in CONTEXT NODES — never invent IDs.
      - For add_node and modify_node: provide complete, runnable Python code.
      - Call submit when all required mutations have been recorded.
      - Respond with ONLY the JSON object, nothing else.
    """
).strip()


# ---------------------------------------------------------------------------
# Remote environment client
# ---------------------------------------------------------------------------

class ASTEnvClient(EnvClient[ASTAction, ASTObservation, ASTState]):
    """WebSocket client for the GraphForge AST environment server."""

    def _step_payload(self, action: ASTAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ASTObservation]:
        obs = ASTObservation.model_validate(payload)
        return StepResult(
            observation=obs,
            reward=obs.reward if isinstance(obs.reward, (int, float)) else None,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ASTState:
        return ASTState.model_validate(payload)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_divider(char: str = "-", width: int = 60) -> None:
    print(char * width, flush=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _is_valid_python(code: str) -> bool:
    try:
        _ast.parse(code)
        return True
    except SyntaxError:
        return False


def compute_code_correctness(mutations: List[dict]) -> float:
    code_mutations = [m for m in mutations if m.get("code")]
    if not code_mutations:
        return 0.0
    valid = sum(1 for m in code_mutations if _is_valid_python(m["code"]))
    return valid / len(code_mutations)


def compute_sequence_score(mutations: List[dict], expected_actions: List[dict]) -> float:
    if not expected_actions:
        return 1.0
    n_expected = len(expected_actions)
    n_compare = min(n_expected, len(mutations))
    matches = 0.0
    for i in range(n_compare):
        exp = expected_actions[i]
        act = mutations[i]
        if act.get("type", "").upper() == exp.get("type", "").upper():
            matches += 0.5
        if act.get("node_id", "") == exp.get("node_id", ""):
            matches += 0.5
    return matches / n_expected


def compute_score(
    obs: ASTObservation,
    query_obj: dict,
) -> Tuple[float, float, float]:
    mutations = obs.mutations_so_far
    if not mutations:
        return 0.0, 0.0, 0.0
    code_score = compute_code_correctness(mutations)
    expected = (query_obj.get("llm_result") or {}).get("actions") or []
    seq_score = compute_sequence_score(mutations, expected)
    total = 0.5 * code_score + 0.5 * seq_score
    return total, code_score, seq_score


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_observation(obs: ASTObservation, last_error: Optional[str] = None) -> str:
    lines = [
        f"TASK: {obs.query}",
        f"CENTRAL NODE: {obs.central_node}",
        "",
        f"CONTEXT NODES ({len(obs.context_nodes)}):",
        json.dumps(list(obs.context_nodes.keys()), indent=2),
        "",
        "NODE DETAILS:",
        json.dumps(obs.context_nodes, indent=2),
    ]
    if obs.not_found_nodes:
        lines.append(f"\nNOT FOUND (do not request again): {json.dumps(obs.not_found_nodes)}")
    if obs.mutations_so_far:
        recorded = [
            {"type": m.get("type"), "node_id": m.get("node_id")}
            for m in obs.mutations_so_far
        ]
        lines.append(
            f"\nMUTATIONS RECORDED ({len(obs.mutations_so_far)}): {json.dumps(recorded)}"
        )
    if last_error:
        lines.append(f"\nLAST ACTION ERROR: {last_error}. Try a different action.")
    lines.append("\nRespond with ONE JSON action object:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing + fallback
# ---------------------------------------------------------------------------

def _build_action(data: dict) -> Optional[ASTAction]:
    action_type = data.get("action", "")
    valid = {"get_neighbors", "add_node", "modify_node", "delete_node", "submit"}
    if action_type not in valid:
        return None
    params = {k: v for k, v in data.items() if k != "action"}
    return ASTAction(action_type=action_type, parameters=params)  # type: ignore[arg-type]


def parse_action(text: str) -> Optional[ASTAction]:
    stripped = text.strip()
    try:
        return _build_action(json.loads(stripped))
    except (json.JSONDecodeError, Exception):
        pass

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i, ch in enumerate(stripped[start:], start):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _build_action(json.loads(stripped[start : i + 1]))
                except (json.JSONDecodeError, Exception):
                    return None

    return None


def fallback_action() -> ASTAction:
    return ASTAction(action_type="submit", parameters={})


def get_llm_action(
    client: OpenAI,
    obs: ASTObservation,
    last_error: Optional[str] = None,
) -> Tuple[Optional[ASTAction], Optional[str]]:
    prompt = format_observation(obs, last_error=last_error)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        text = (response.choices[0].message.content or "").strip()
        action = parse_action(text)
        if action is None:
            return None, f"parse_fail:{text[:80]!r}"
        return action, None
    except Exception as exc:
        return None, f"llm_error:{exc}"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(env, client: OpenAI, task_name: str, seed: int) -> float:
    """Run one full episode against the remote env. Returns final score in [0, 1]."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    reset_result: StepResult[ASTObservation] = env.reset(seed=seed)
    obs = reset_result.observation

    # Fetch full server state to get query_obj (needed for reference scoring).
    server_state: ASTState = env.state()
    query_obj = server_state.query_obj

    expected_actions = (query_obj.get("llm_result") or {}).get("actions") or []
    complexity = query_obj.get("complexity", "unknown")

    log_divider("=")
    log_info(f"query    : {obs.query}")
    log_info(f"central  : {obs.central_node}")
    log_info(f"complexity: {complexity}")
    log_info(f"context  : {len(obs.context_nodes)} nodes, {len(obs.context_edges)} edges")
    log_info(f"reference: {len(expected_actions)} expected mutation(s)")
    if expected_actions:
        for i, a in enumerate(expected_actions, 1):
            log_info(f"  ref[{i}] {a.get('type'):<8} {a.get('node_id')}")
    log_divider()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    code_score = 0.0
    seq_score = 0.0
    success = False
    last_error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            if obs.done:
                break

            log_info(
                f"step={step} context={len(obs.context_nodes)} nodes  "
                f"mutations_so_far={len(obs.mutations_so_far)}"
            )

            action, err = get_llm_action(client, obs, last_error=last_error)

            if action is None:
                log_info(f"  llm parse failed — using fallback submit  ({err})")
                action = fallback_action()
                last_error = err
            else:
                last_error = None

            prev_node_count = len(obs.context_nodes)
            step_result: StepResult[ASTObservation] = env.step(action)
            obs = step_result.observation

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            at = action.action_type
            node = action.parameters.get("node_id", "")
            action_str = f"{at}({node})" if node else f"{at}()"

            log_step(step=step, action=action_str, reward=reward, done=done, error=err)

            if at == "get_neighbors":
                added = len(obs.context_nodes) - prev_node_count
                log_info(f"  get_neighbors → +{added} nodes (context now {len(obs.context_nodes)})")
            elif at in ("add_node", "modify_node"):
                code = action.parameters.get("code", "")
                valid = _is_valid_python(code) if code else None
                validity_str = "valid ✓" if valid else ("invalid ✗" if valid is False else "no-code")
                log_info(f"  {at} → {node}  code={validity_str}")
                if code and not valid:
                    log_info(f"  [!] syntax error in submitted code")
            elif at == "delete_node":
                log_info(f"  delete_node → {node}")
            elif at == "submit":
                log_info(f"  submit → episode ending with {len(obs.mutations_so_far)} mutation(s)")

            rewards.append(reward)
            steps_taken = step

            if done:
                break

        score, code_score, seq_score = compute_score(obs, query_obj)
        success = score >= SUCCESS_SCORE_THRESHOLD

        log_divider()
        log_info(f"episode done  steps={steps_taken}  mutations={len(obs.mutations_so_far)}")
        log_info(f"score breakdown:")
        log_info(
            f"  code_correctness = {code_score:.2f}  "
            f"({sum(1 for m in obs.mutations_so_far if m.get('code') and _is_valid_python(m['code']))}"
            f"/{sum(1 for m in obs.mutations_so_far if m.get('code'))} valid)"
        )
        log_info(
            f"  sequence_match   = {seq_score:.2f}  "
            f"(ref={len(expected_actions)} actions, agent={len(obs.mutations_so_far)} mutations)"
        )
        log_info(f"  total_score      = {score:.2f}  success={str(success).lower()}")

        if obs.mutations_so_far:
            log_info("agent mutations:")
            for i, m in enumerate(obs.mutations_so_far, 1):
                code = m.get("code", "")
                valid_str = f"  code={'valid ✓' if _is_valid_python(code) else 'invalid ✗'}" if code else ""
                log_info(f"  [{i}] {m.get('type'):<8} {m.get('node_id')}{valid_str}")

        if expected_actions:
            log_info("reference mutations:")
            for i, a in enumerate(expected_actions, 1):
                match_type = (
                    len(obs.mutations_so_far) > i - 1
                    and obs.mutations_so_far[i - 1].get("type", "").upper()
                    == a.get("type", "").upper()
                )
                match_node = (
                    len(obs.mutations_so_far) > i - 1
                    and obs.mutations_so_far[i - 1].get("node_id") == a.get("node_id")
                )
                log_info(
                    f"  [{i}] {a.get('type'):<8} {a.get('node_id')}"
                    f"  type={'✓' if match_type else '✗'}  node={'✓' if match_node else '✗'}"
                )

        log_divider("=")

    except Exception as exc:
        print(f"[DEBUG] episode exception: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")

    async_env = ASTEnvClient(base_url=HF_SPACE_URL)
    env = async_env.sync()

    tasks = [
        ("low_complexity",    10),
        ("medium_complexity", 20),
        ("high_complexity",   30),
    ]
    scores: dict[str, float] = {}

    log_info(f"Connecting to HF Space: {HF_SPACE_URL}")

    with env:
        for task_name, seed in tasks:
            try:
                scores[task_name] = run_task(env, llm_client, task_name, seed)
            except Exception as exc:
                print(f"[DEBUG] task={task_name} failed: {exc}", flush=True)
                scores[task_name] = 0.0

    log_divider("=")
    print("\n=== FINAL SCORES ===", flush=True)
    for task, s in scores.items():
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {task:<22} {bar}  {s:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(
        f"  {'Average':<22} {'█' * int(avg * 20) + '░' * (20 - int(avg * 20))}  {avg:.4f}",
        flush=True,
    )
    log_divider("=")


if __name__ == "__main__":
    main()

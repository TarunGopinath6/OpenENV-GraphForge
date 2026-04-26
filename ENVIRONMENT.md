# GraphForge Environment

GraphForge is an OpenENV-compliant reinforcement-learning environment in which an agent incrementally constructs a typed Python call graph to satisfy a code-architecture task. The agent builds up a graph of modules and functions, connects them with typed edges, optionally attaches body templates, then materializes the graph to real Python source files and submits for scoring.

---

## Overview

At a high level, each episode looks like this:

```
reset(tier, task_id?) → GraphObservation
    ↓
step(GraphAction) → GraphObservation    (repeat up to turn_cap times)
    ↓
submit → terminal GraphObservation with final reward
```

The environment enforces two hard budget limits: a **turn cap** and a **token budget**. When either is exhausted, the episode ends and a `_forced_terminal_reward` is computed — materializing fresh if no cache exists, then delegating to `terminal_reward` with whatever test results and mypy state are available. Agents that exhaust their budget without submitting still receive a meaningful gradient signal rather than silence.

---

## State

### `GraphForgeState` (internal, not directly exposed)

The environment maintains one `GraphForgeState` per live episode. Its key fields:

| Field                   | Type                               | Description                                                                                                                               |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `graph`                 | `GraphState`                       | The graph being built: modules, nodes, edges                                                                                              |
| `task_spec`             | `TaskSpec`                         | The task to solve (constraints + behavioral tests)                                                                                        |
| `turn_cap`              | `int`                              | Maximum steps allowed                                                                                                                     |
| `token_budget`          | `int`                              | Maximum tokens (action + observation) allowed                                                                                             |
| `tokens_used`           | `int`                              | Running total of tokens consumed                                                                                                          |
| `cumulative_reward`     | `float`                            | Running reward sum (per-turn penalties accumulate here)                                                                                   |
| `is_terminal`           | `bool`                             | Set to `True` when the episode ends                                                                                                       |
| `materialization_cache` | `dict[str, str] \| None`           | Python source per module from the last `materialize_and_validate` call; cleared to `None` by the dispatcher after any successful mutation |
| `last_test_results`     | `list[TestResult] \| None`         | Results from the last `run_behavioral_tests` call                                                                                         |
| `action_history`        | `list[(action_type, params_json)]` | Used to detect repeat actions                                                                                                             |

### `GraphState` (the graph itself)

The graph has three collections:

- **modules** — `GraphModule(name, responsibility)`: declared Python files, each with a responsibility tag (`io`, `transform`, `validation`, `orchestration`, …)
- **nodes** — `GraphNode(name, module, signature, purity, error_policy, body_template, …)`: individual functions, each belonging to exactly one module
- **edges** — `GraphEdge(caller, callee, arg_mapping)`: directed call-edges between fully-qualified function names (`module.func → module.func`)

---

## Inputs (Actions)

Every action is a `GraphAction(action_type, parameters)`. There are 14 action types split into two groups.

### Mutation Actions

These change the graph. Failed mutations are rolled back — the graph is snapshot-copied before every dispatch and restored if `ActionResult.success` is `False`.

| Action            | Key Parameters                                          | What It Does                                                                               |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `add_module`      | `name`, `responsibility`                                | Declares a new module                                                                      |
| `remove_module`   | `name`                                                  | Removes a module (fails if it still has nodes)                                             |
| `add_node`        | `name`, `module`, `signature`, `purity`, `error_policy` | Adds a function to a module; signature is parsed by `TypeEngine`                           |
| `remove_node`     | `name`, `module`                                        | Removes a function (fails if it still has edges)                                           |
| `set_node_module` | `name`, `current_module`, `new_module`                  | Moves a function to a different module; all edges referencing it are updated automatically |
| `attach_body`     | `name`, `module`, `template`, `args`                    | Attaches a body template to a function; template is validated and dry-run rendered         |
| `add_edge`        | `caller`, `callee`, `arg_mapping`                       | Adds a call edge; validates type compatibility and checks for import cycles                |
| `remove_edge`     | `caller`, `callee`                                      | Removes an edge                                                                            |

### Information / Control Actions

These do not mutate the graph (no rollback needed).

| Action                     | Key Parameters     | What It Does                                                                                       |
| -------------------------- | ------------------ | -------------------------------------------------------------------------------------------------- |
| `query_spec`               | `constraint_kind?` | Returns constraint satisfaction status for all visible constraints                                 |
| `query_subgraph`           | `scope`            | Returns nodes/edges for a `module:<name>`, `neighbors:<node>`, or `path:<from>:<to>` scope         |
| `query_types`              | `scope`            | Returns parameter types and return types; flags `Any` contamination and type-consistency errors    |
| `materialize_and_validate` | —                  | Converts the graph to Python source files and runs mypy; populates `materialization_cache`         |
| `run_behavioral_tests`     | —                  | Runs pytest behavioral tests against the cached source (requires prior successful materialization) |
| `submit`                   | —                  | Ends the episode; triggers final scoring against all constraints (including hidden ones)           |

---

## Processes (Engine Components)

The environment delegates all heavy lifting to eight engine components wired together by `ActionDispatcher`.

### `TypeEngine`

- Parses function signatures from strings (`(x: int, y: str) -> bool`)
- Validates that edge `arg_mapping` entries are type-compatible between caller and callee
- Detects `Any` contamination in node signatures
- Checks global type consistency across all edges
- Builds the import-dependency graph and detects cycles using DFS

### `BodyTemplateLibrary`

- Stores named code-generation templates (e.g. `identity`, `return_none`, `raise_not_implemented`)
- Validates template arguments before attachment
- Renders templates to Python source strings at materialize time

### `Materializer`

- Converts `GraphState → dict[module_name, python_source]`
- For each module, collects its nodes (sorted by `decl_order`), computes the needed cross-module imports from the edge set, and generates function definitions
- Functions with a `body_template` get the rendered body; all others get `raise NotImplementedError`
- On any generation error, marks `MaterializeResult.success = False`

### `Validator`

- Takes the `dict[module_name, python_source]` produced by `Materializer`
- Runs `ast.parse` on each file (parse errors)
- Runs `mypy` in a subprocess on the combined source (type errors)
- Returns `ValidationResult(parse_errors, mypy_errors)`

### `BehavioralTestRunner`

- Takes the materialized source and a list of `BehavioralTest` objects
- Each `BehavioralTest` carries a complete pytest file as a string (`hypothesis_source`)
- Writes modules to a temp directory, writes the test file, and runs `pytest` in a subprocess
- Returns `list[TestResult(test_id, passed, error_msg)]`

### `ConstraintChecker`

- Evaluates a list of `ConstraintSpec` objects against a `GraphState`
- Supports 20+ constraint kinds including structural (`node_exists`, `edge_exists`, `module_count`, `acyclic_imports`, `dag_depth_max`, `fan_in_max`, `fan_out_max`, `internal_only`, `module_size_max`), type-level (`return_type`, `arg_type`, `pure_function`, `no_any_types`, `type_consistency`), and behavioral (`error_handling_present`, `error_handling_absent`)
- Constraints marked `hidden=True` are not shown to the agent in observations; they are only evaluated at `submit` time

### `RewardEngine`

- Computes both per-turn incremental rewards and the terminal reward at `submit`

### `TokenCounter`

- Counts tokens consumed by each action and each observation
- The running total in `GraphForgeState.tokens_used` includes both directions

---

## Reward Structure

### Per-Turn (applied every step)

| Event                                   | Delta                        |
| --------------------------------------- | ---------------------------- |
| Base step cost                          | −0.1                         |
| Token cost                              | −0.001 × tokens_this_turn    |
| Successful non-repeat action            | +0.05 (net: −0.05)           |
| Failed mutation                         | −0.5 (net: −0.6)             |
| Repeat action (identical type + params) | −1.0                         |
| Malformed / unknown action              | −2.1 (base + −2.0 MALFORMED) |

### Terminal (applied once at `submit`)

Terminal reward replaces nothing — it adds on top of whatever cumulative reward has accumulated.

| Component                                                   | Value                                                                                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Materialization fails                                       | −8.0 to −4.0 (partial credit: −8.0 + 4.0 × ok_modules/total_modules; fully broken graph = −8.0, one bad module in many ≈ −4.0) |
| Each visible structural constraint satisfied                | +1.0                                                                                                                           |
| All visible structural constraints satisfied                | +5.0 bonus                                                                                                                     |
| Each behavioral test passed                                 | +3.0                                                                                                                           |
| All behavioral tests passed                                 | +5.0 bonus                                                                                                                     |
| mypy passes with zero errors                                | +3.0                                                                                                                           |
| Token efficiency (only if all structural + behavioral pass) | up to +5.0 × fraction of budget remaining                                                                                      |

Hidden constraints are evaluated at submit but their individual results are **never revealed to the agent** — not during the episode, not in the final observation. The `ConstraintSummary` in every observation exposes the _count_ of hidden constraints (`summary.hidden`) and the total count (`summary.total`), but no per-constraint satisfaction status for hidden ones is ever returned. The terminal reward formula operates on **whatever constraints the caller passes**; `_handle_submit` and `_forced_terminal_reward` are responsible for filtering to visible-only before calling `terminal_reward` — the reward engine itself no longer silently strips hidden constraints. Hidden structural constraints are evaluated separately via `check_all` and their results are surfaced in the submit response as `hidden_constraints_satisfied` / `hidden_constraints_total` — analysis-only metadata that does not touch the reward. See the [Hidden Constraints](#hidden-constraints) section for the full breakdown.

---

## Outputs (Observations)

Every call to `step()` or `reset()` returns a `GraphObservation`. Its fields:

| Field                     | Type                     | Description                                                                                                                             |
| ------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `done`                    | `bool`                   | `True` when the episode has ended                                                                                                       |
| `reward`                  | `float \| None`          | Cumulative reward — only populated when `done=True`                                                                                     |
| `episode_id`              | `str`                    | UUID for this episode                                                                                                                   |
| `timestep`                | `int`                    | Current step count                                                                                                                      |
| `turn_budget_remaining`   | `int`                    | Steps left before forced termination                                                                                                    |
| `token_budget_remaining`  | `int`                    | Tokens left before forced termination                                                                                                   |
| `graph_state`             | `GraphState`             | Full current graph (modules, nodes, edges)                                                                                              |
| `last_action_result`      | `ActionResult \| None`   | Success/failure + optional query response for the action just taken                                                                     |
| `constraint_summary`      | `ConstraintSummary`      | Counts: total, visible, hidden, satisfied-visible                                                                                       |
| `visible_constraints`     | `list[ConstraintStatus]` | The non-hidden constraints with per-constraint satisfaction status (`ConstraintStatus` extends `ConstraintSpec` with `satisfied: bool`) |
| `repeated_action_warning` | `bool`                   | `True` when the most recent action was an exact repeat (type + params match); repeat penalty already applied                            |
| `available_actions`       | `list[str]`              | Current valid action types (`run_behavioral_tests` is absent until `materialize_and_validate` succeeds)                                 |
| `task_description`        | `str`                    | Natural-language description of what the task requires                                                                                  |
| `metadata`                | `dict`                   | `tokens_used` and `step_count`                                                                                                          |

---

## Tasks

Tasks are drawn from a `TaskBank` of ~52 pre-built specs across three tiers. A task is sampled at `reset()` time (or specified by `task_id`).

### Tiers

| Tier | Nodes | Modules | Constraints | Hidden  | Behavioral Tests | Turn Cap | Token Budget |
| ---- | ----- | ------- | ----------- | ------- | ---------------- | -------- | ------------ |
| 1    | 5–8   | 1–2     | ~15         | ~30%    | 0–2              | 20       | 4,000        |
| 2    | 10–18 | 3–4     | ~30         | ~35%    | 3–5              | 35       | 8,000        |
| 3    | 18–30 | 5–7     | ~50         | ~24–27% | 5–8              | 55       | 14,000       |

### Tier 1 Templates (4 templates × 4 variants = 16 tasks)

- **simple_pipeline** — single-module function utilities (numbers, strings, lists, math)
- **validator_chain** — single-module boolean validators (email, phone, url, password)
- **transform_reduce** — single-module map-filter-reduce (sales, logs, inventory, scores)
- **config_loader** — single-module config loading with defaults and validation

Variants 0/1 differ cosmetically (domain swap only). Variants 2/3 require at least one `edge_exists` constraint (variant 3 requires two), so a one-pass node-dump that solves v0/v1 will not satisfy v2/v3.

### Tier 2 Templates (6 templates × 4 variants = 24 tasks)

- **data_pipeline** — 3-module ETL (csv, json, xml, tsv)
- **validator_chain** — 3-module validation with parser + rules + reporter
- **event_router** — 3-module event routing with ingestion + handlers + output
- **config_loader** — 3-module config system with reader + schema + assembler
- **batch_processor** — 3-module batch processing with reader + processor + writer
- **transform_reduce** — 3-module analytics with ingest + compute + output

### Tier 3 Templates (4 templates × 3 variants = 12 tasks)

- **microservice** — 5 modules: auth, validation, core, storage, api
- **query_engine** — 6 modules: lexer, parser, planner, optimizer, executor, formatter
- **workflow_engine** — 5 modules: definition, scheduler, executor, state, monitor
- **plugin_system** — 7 modules: discovery, loader, validator, registry, lifecycle, pluginapi, events

---

## Hidden Constraints

Hidden constraints (`ConstraintSpec.hidden = True`) are a core design element of the environment. They exist to prevent the agent from over-fitting to the exact scoring criteria — the agent is told _what_ to build but not every rule it will be graded on.

### What the Agent Can and Cannot See

| Information                                           | Visible to Agent?                            |
| ----------------------------------------------------- | -------------------------------------------- |
| Total constraint count (visible + hidden)             | Yes — `constraint_summary.total`             |
| Count of hidden constraints                           | Yes — `constraint_summary.hidden`            |
| Count of visible constraints satisfied                | Yes — `constraint_summary.satisfied_visible` |
| Which specific constraints are hidden                 | No                                           |
| Whether any individual hidden constraint is satisfied | No — ever, including after submit            |
| Hidden constraint results in the final observation    | No                                           |

The `visible_constraints` list in every observation contains only non-hidden constraints. The agent can call `query_spec` to check visible constraint satisfaction at any time, but there is no action that exposes hidden constraint results.

### How Hidden Constraints Interact with Reward

Hidden constraints do **not** add to or subtract from the terminal reward. `RewardEngine.terminal_reward` treats its `constraints` argument as authoritative — it applies whatever is passed without re-filtering. The callers (`_handle_submit` and `_forced_terminal_reward`) are responsible for passing only visible constraints. Hidden constraints are evaluated separately and reported as metadata only. This means:

- Satisfying a hidden structural constraint: **no reward effect**
- Violating a hidden structural constraint: **no reward effect**
- The all-structural bonus (+5.0) is gated only on visible constraints all being satisfied
- Behavioral tests and mypy are not subject to the hidden filter — all tests and the type-check bonus apply regardless

Hidden constraints exist as an **evaluation signal** for benchmarking and analysis, not as a training signal for the agent. Their aggregate satisfaction (`hidden_constraints_satisfied` / `hidden_constraints_total`) is surfaced in the submit response for offline analysis.

### Complete Constraint Kind Reference

Every constraint kind supported by `ConstraintChecker`, with its typical visibility pattern across the task bank:

#### Structural — typically **visible**

| Kind                     | Target           | Value              | Description                                                |
| ------------------------ | ---------------- | ------------------ | ---------------------------------------------------------- |
| `node_exists`            | `module.func`    | —                  | Function must exist in the graph                           |
| `module_count`           | —                | `int`              | Total number of modules must equal value                   |
| `module_count_range`     | —                | `[min, max]`       | Module count must be within range                          |
| `module_responsibility`  | module name      | responsibility tag | Module's `responsibility` field must match                 |
| `acyclic_imports`        | —                | —                  | Import dependency graph must be a DAG (no cycles)          |
| `entrypoint_exists`      | `module.func`    | —                  | Function must exist (marks it as a public entry point)     |
| `pure_function`          | `module.func`    | —                  | Function's `purity` field must be `"pure"`                 |
| `return_type`            | `module.func`    | type string        | Function's declared return type must match                 |
| `error_handling_present` | `module.func`    | —                  | `error_policy` must be `"return_none"` or `"return_error"` |
| `fan_in_max`             | `module.func`    | `int`              | Number of incoming call-edges must be ≤ value              |
| `edge_exists`            | `caller::callee` | —                  | A specific call-edge must be present (primary flow edges)  |

#### Structural — typically **hidden**

| Kind                                 | Target           | Value              | Description                                                                        |
| ------------------------------------ | ---------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `no_any_types`                       | —                | —                  | No function anywhere in the graph may use `Any` in its signature                   |
| `type_consistency`                   | —                | —                  | All call-edges must be type-compatible (no mismatched arg→param types)             |
| `dag_depth_max`                      | —                | `int`              | Longest path through the call graph must be ≤ value                                |
| `fan_out_max`                        | `module.func`    | `int`              | Number of outgoing call-edges from a function must be ≤ value                      |
| `internal_only`                      | `module.func`    | —                  | All callers of this function must be in the same module                            |
| `module_size_max`                    | module name      | `int`              | Number of nodes in this module must be ≤ value                                     |
| `error_handling_absent`              | `module.func`    | —                  | `error_policy` must be `"raise"` (pure functions must not silently swallow errors) |
| `edge_exists` (cross-module)         | `caller::callee` | —                  | Specific cross-module call edges in the expected data flow direction               |
| `pure_function` (secondary)          | `module.func`    | —                  | Secondary/helper functions that must also be declared pure                         |
| `return_type` (secondary)            | `module.func`    | type string        | Return type constraints on non-entrypoint functions                                |
| `error_handling_present` (secondary) | `module.func`    | —                  | Error policy requirements on non-primary functions                                 |
| `module_responsibility` (secondary)  | module name      | responsibility tag | Responsibility tag requirements on non-primary modules                             |
| `node_absent`                        | `module.func`    | —                  | A function must NOT exist (anti-patterns to avoid)                                 |
| `edge_absent`                        | `caller::callee` | —                  | A specific call-edge must NOT be present                                           |

#### Type/Signature — available, mostly **hidden**

| Kind                | Target        | Value                      | Description                                            |
| ------------------- | ------------- | -------------------------- | ------------------------------------------------------ |
| `signature_matches` | `module.func` | regex                      | The full signature string must match a regex pattern   |
| `arg_type`          | `module.func` | `[position_or_name, type]` | A specific parameter must have a given type annotation |

#### Behavioral — only evaluated when materialization cache is available

These four kinds are short-circuited to `False` in the plain `check_one` path; they require `check_all_with_cache` with externally-computed results passed in.

| Kind                     | Target | Value   | Description                                                 |
| ------------------------ | ------ | ------- | ----------------------------------------------------------- |
| `materializes`           | —      | —       | The graph must produce error-free Python source             |
| `imports_resolve`        | —      | —       | All cross-module imports must resolve without `ImportError` |
| `type_checks`            | —      | —       | mypy must report zero errors                                |
| `behavioral_test_passes` | —      | test_id | A specific named behavioral test must pass                  |

### Tier-by-Tier Hidden Constraint Breakdown

The fraction of constraints that are hidden increases with tier complexity.

**Tier 1 (~30% hidden, ~5 hidden per task)**

Visible: `node_exists` for all named functions, `module_count`, `acyclic_imports`, primary `module_responsibility`, primary `pure_function`, primary `return_type`, primary `error_handling_present`, `dag_depth_max` (simple_pipeline), `fan_in_max` (validator_chain), primary `edge_exists` (transform_reduce), `entrypoint_exists` (config_loader)

Hidden: `no_any_types`, secondary `pure_function`, secondary `return_type`, `fan_out_max`, secondary `error_handling_present`, secondary `dag_depth_max`, secondary `edge_exists`

**Tier 2 (~35% hidden, ~10–11 hidden per task)**

Visible: All Tier 1 visible kinds, plus primary `edge_exists` between modules (for some templates), primary `module_responsibility` for I/O modules

Hidden: All Tier 1 hidden kinds, plus `type_consistency`, cross-module `edge_exists` (secondary data-flow edges), secondary `module_responsibility` (e.g. the middle `transform` module), `internal_only`, `module_size_max`, secondary `error_handling_present` on output-side functions

**Tier 3 (~24–27% hidden, ~12–14 hidden per task)**

Visible: Core structural skeleton — which modules exist, all module responsibility tags (`module_responsibility` for every module is now visible so agents always know what each module is for), main entrypoint, primary `pure_function`, primary `return_type`, primary `error_handling_present`, `module_size_max` for one module, `no_any_types` (basic coding hygiene), critical-path `edge_exists` (the entry-point chain defining the architecture)

Hidden: `type_consistency`, `dag_depth_max`, `fan_in_max`/`fan_out_max`, `internal_only`, `error_handling_absent` on pure functions — all legitimately discovery-through-iteration constraints

---

## Episode Lifecycle

```
1. reset()
   ├── Sample or load TaskSpec from TaskBank
   ├── Initialize empty GraphState
   └── Return initial GraphObservation (graph is empty, task_description is set)

2. step() × N  (agent loop)
   ├── Count action tokens → add to tokens_used
   ├── Validate action_type (unknown → malformed penalty, terminal obs)
   ├── Check for repeat action (exact type + params match in action_history)
   ├── Dispatch to handler
   │   ├── Snapshot graph for rollback
   │   ├── Run handler (mutation or query)
   │   └── Restore snapshot if handler returns success=False
   ├── Accumulate per-turn reward
   ├── Increment step_count
   ├── Check turn_cap → if reached: run _forced_terminal_reward, mark terminal
   ├── Check token_budget → if exceeded: run _forced_terminal_reward, mark terminal
   ├── Build GraphObservation (includes updated graph, constraint summary, result)
   └── Count observation tokens → add to tokens_used

3. submit (via step with action_type="submit")
   ├── Always materialize fresh (ignores any existing cache)
   ├── Run mypy on fresh materialized source
   ├── Run behavioral tests
   ├── Split constraints: visible → terminal_reward; hidden → check_all (metadata only)
   ├── Compute terminal reward (visible constraints only)
   ├── Add hidden_constraints_satisfied / hidden_constraints_total to query_response
   ├── Set is_terminal = True
   └── Return final GraphObservation with done=True and reward=cumulative
```

---

## Concurrent Sessions

`GraphForgeEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True`. Each call to `reset()` reinitializes `_state` in place, so a single environment instance supports sequential episodes but not truly concurrent ones. For parallelism, instantiate multiple `GraphForgeEnvironment` objects.

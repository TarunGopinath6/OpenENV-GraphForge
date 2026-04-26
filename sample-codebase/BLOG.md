# OpenENV-GraphForge

**A graph-first code-editing RL environment for Python repositories, built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).**

## 1. Problem

Current code-generating LMs emit source code token-by-token with no structural awareness. This fails for multi-step program construction because:

- **Token bloat:** By turn 30 of a non-trivial task, a small model burns most of its context on already-written code, not planning.
- **Implicit structure:** "Which functions call this one?" requires re-parsing every file. These are O(1) on a typed graph, O(N) on text.
- **Deferred error signal:** A wrong implementation propagates silently until the full program is run.

## What's Our Fix?

**GraphForge** inverts the pipeline. The agent mutates a typed function-call Directed Acyclic Graph (DAG) with the nodes being modules, classes, functions, and external dependencies.

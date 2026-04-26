"""Validator: parse, import-resolution, and mypy checks on materialized source."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from models import MaterializeResult


class Validator:
    """Runs three levels of validation on materialized Python source."""

    def __init__(self) -> None:
        self._mypy_cache: Dict[str, Tuple[bool, str]] = {}

    def parse_check(self, module_sources: Dict[str, str]) -> List[str]:
        """Return syntax error strings for any source that fails ast.parse."""
        errors: List[str] = []
        for module_name, source in module_sources.items():
            try:
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"Module {module_name!r}: SyntaxError: {e}")
        return errors

    def import_resolution_check(self, module_sources: Dict[str, str]) -> List[str]:
        """Verify every inter-module import references a function that exists."""
        errors: List[str] = []
        module_exports: Dict[str, set] = {}

        for module_name, source in module_sources.items():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            funcs = {
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            }
            module_exports[module_name] = funcs

        for module_name, source in module_sources.items():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    src_mod = node.module
                    if src_mod not in module_exports:
                        continue  # external stdlib/third-party, skip
                    for alias in node.names:
                        name = alias.name
                        if name != "*" and name not in module_exports[src_mod]:
                            errors.append(
                                f"Module {module_name!r}: cannot import {name!r} "
                                f"from {src_mod!r} (not defined)"
                            )
        return errors

    def mypy_check(
        self, module_sources: Dict[str, str], strict: bool = True
    ) -> Tuple[bool, str]:
        """Write sources to a temp dir, run mypy, clean up.

        Returns (passed, output_text).
        """
        cache_key = self._make_cache_key(module_sources, strict)
        if cache_key in self._mypy_cache:
            return self._mypy_cache[cache_key]

        result = self._run_mypy(module_sources, strict)
        self._mypy_cache[cache_key] = result
        return result

    def _run_mypy(
        self, module_sources: Dict[str, str], strict: bool
    ) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix="graphforge_mypy_") as tmpdir:
            tmp = Path(tmpdir)
            for module_name, source in module_sources.items():
                (tmp / f"{module_name}.py").write_text(source, encoding="utf-8")

            cmd = [sys.executable, "-m", "mypy"]
            if strict:
                cmd.append("--strict")
            cmd += ["--ignore-missing-imports", "--no-error-summary", str(tmp)]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                output = proc.stdout + proc.stderr
                passed = proc.returncode == 0
            except subprocess.TimeoutExpired:
                output = "mypy check timed out"
                passed = False
            except FileNotFoundError:
                output = "mypy not found"
                passed = False

        return passed, output

    def full_validate(
        self, module_sources: Dict[str, str], strict: bool = True
    ) -> MaterializeResult:
        """Run all three checks in sequence.

        Order: parse → import resolution → mypy. A parse failure short-circuits
        immediately — mypy is never invoked on unparseable source. mypy runs on
        the full temp directory (package scope), not per-file, so cross-module
        import resolution works correctly.
        """
        parse_errors = self.parse_check(module_sources)
        if parse_errors:
            return MaterializeResult(
                success=False,
                module_sources=module_sources,
                parse_errors=parse_errors,
                mypy_errors=[],
            )

        import_errors = self.import_resolution_check(module_sources)
        mypy_ok, mypy_output = self.mypy_check(module_sources, strict=strict)
        mypy_errors = [] if mypy_ok else [mypy_output]

        success = len(import_errors) == 0 and mypy_ok
        return MaterializeResult(
            success=success,
            module_sources=module_sources,
            parse_errors=import_errors,
            mypy_errors=mypy_errors,
        )

    def _make_cache_key(self, module_sources: Dict[str, str], strict: bool) -> str:
        combined = str(strict) + "".join(
            f"{k}:{v}" for k, v in sorted(module_sources.items())
        )
        return hashlib.md5(combined.encode()).hexdigest()

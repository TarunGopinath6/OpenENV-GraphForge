"""Behavioral test runner: runs Hypothesis-based tests in a sandboxed subprocess."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from models import BehavioralTest, TestResult


class BehavioralTestRunner:
    """Executes static behavioral tests against materialized Python source."""

    def run_tests(
        self,
        module_sources: Dict[str, str],
        behavioral_tests: List[BehavioralTest],
        timeout_s: float = 30.0,
    ) -> List[TestResult]:
        if not behavioral_tests:
            return []

        results: List[TestResult] = []
        for test in behavioral_tests:
            result = self._run_single_test(module_sources, test, timeout_s)
            results.append(result)
        return results

    def _run_single_test(
        self,
        module_sources: Dict[str, str],
        test: BehavioralTest,
        timeout_s: float,
    ) -> TestResult:
        with tempfile.TemporaryDirectory(prefix="graphforge_test_") as tmpdir:
            sandbox = Path(tmpdir)
            self._write_sandbox(sandbox, module_sources, test.hypothesis_source)

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(sandbox / "test_behavior.py"),
                "--tb=short",
                "-q",
                "--no-header",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=str(sandbox),
                )
                output = proc.stdout + proc.stderr
                passed = proc.returncode == 0
                error_msg = None if passed else output[:1000]
            except subprocess.TimeoutExpired:
                passed = False
                error_msg = f"Test {test.test_id!r} timed out after {timeout_s}s"
            except Exception as e:
                passed = False
                error_msg = str(e)

        return TestResult(test_id=test.test_id, passed=passed, error_msg=error_msg)

    def _write_sandbox(
        self,
        sandbox: Path,
        module_sources: Dict[str, str],
        test_source: str,
    ) -> None:
        # Write module files
        for module_name, source in module_sources.items():
            (sandbox / f"{module_name}.py").write_text(source, encoding="utf-8")

        # Write test file
        (sandbox / "test_behavior.py").write_text(test_source, encoding="utf-8")

        # Write conftest to add sandbox to sys.path
        conftest = "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        (sandbox / "conftest.py").write_text(conftest, encoding="utf-8")

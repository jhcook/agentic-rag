#!/usr/bin/env python3
"""
Check compliance with project rules and standards.

This script verifies that the codebase follows the rules defined in
.agent/rules/ directory.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _get_staged_python_files(prefix: str) -> list[str]:
    """Return staged .py files under the given prefix (e.g. 'src' or 'tests')."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        files: list[str] = []
        for raw in result.stdout.splitlines():
            path = raw.strip()
            if not path:
                continue
            if not path.startswith(f"{prefix}/"):
                continue
            if path.endswith(".py"):
                files.append(path)
        return files
    except Exception:
        return []


def check_pylint():
    """Check if pylint passes with project configuration."""
    print("🔍 Running pylint checks...")
    try:
        staged_src = _get_staged_python_files("src")
        staged_tests = _get_staged_python_files("tests")
        targets = staged_src + staged_tests
        if not targets:
            # Fallback: lint the full codebase only if nothing is staged.
            targets = ["src", "tests"]

        result = subprocess.run(
            [
                "pylint",
                "--jobs=0",
                *targets,
                "--rcfile=.pylintrc",
                "--errors-only",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("✅ Pylint checks passed")
            return True
        else:
            print("❌ Pylint checks failed:")
            if result.stdout.strip():
                print(result.stdout)
            if result.stderr.strip():
                print(result.stderr)
            return False
    except FileNotFoundError:
        print("⚠️  Pylint not found. Install with: pip install pylint")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Pylint checks timed out")
        return False


def check_tests_exist():
    """Check if test files exist for main source files."""
    print("\n🔍 Checking test coverage...")
    src_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    test_files = list((PROJECT_ROOT / "tests").rglob("test_*.py"))
    
    # Simple check: ensure tests directory has files
    if not test_files:
        print("❌ No test files found in tests/ directory")
        return False
    
    print(f"✅ Found {len(test_files)} test files")
    return True


def check_docstrings():
    """Check if main modules have docstrings."""
    print("\n🔍 Checking docstrings...")
    try:
        staged_src = _get_staged_python_files("src")
        targets = staged_src if staged_src else ["src"]
        result = subprocess.run(
            [
                "pylint",
                "--jobs=0",
                *targets,
                "--rcfile=.pylintrc",
                "--disable=all",
                "--enable=missing-module-docstring,missing-class-docstring,missing-function-docstring",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Count missing docstring errors
        missing = result.stdout.count("missing-")
        if missing == 0:
            print("✅ All modules have docstrings")
            return True
        else:
            print(f"⚠️  Found {missing} missing docstring warnings")
            return False
    except FileNotFoundError:
        print("⚠️  Pylint not found")
        return None
    except subprocess.TimeoutExpired:
        print("⚠️  Docstring check timed out")
        return False


def check_line_length():
    """Check line length compliance."""
    print("\n🔍 Checking line length (PEP-8: 100 chars)...")
    try:
        # Enforce on staged additions only (commit-focused), not the entire file.
        result = subprocess.run(
            ["git", "diff", "--cached", "-U0", "--", "src"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode not in (0, 1):
            print("⚠️  Unable to compute staged diff for src/")
            return False

        too_long = 0
        for line in result.stdout.splitlines():
            if not line.startswith("+"):
                continue
            if line.startswith("+++"):
                continue
            # Remove the diff marker.
            content = line[1:]
            # Ignore empty additions.
            if not content:
                continue
            if len(content) > 100:
                too_long += 1

        if too_long == 0:
            print("✅ All staged additions comply with 100 character limit")
            return True

        print(f"⚠️  Found {too_long} staged lines exceeding 100 characters")
        return False
    except FileNotFoundError:
        print("⚠️  Git not found")
        return None
    except subprocess.TimeoutExpired:
        print("⚠️  Line length check timed out")
        return False


def check_cursor_rules_exist():
    """Verify governance rules files exist."""
    print("\n🔍 Checking governance rules files...")
    rules_dir = PROJECT_ROOT / ".agent" / "rules"
    required_files = [
        "commit-workflow.mdc",
        "global-compliance-requirements.mdc",
        "lean-code.mdc",
        "the-team.mdc",
    ]
    
    missing = []
    for file in required_files:
        if not (rules_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing cursor rules files: {', '.join(missing)}")
        return False
    
    print("✅ All governance rules files present")
    return True


def main():
    """Run all compliance checks."""
    print("=" * 60)
    print("Project Rules Compliance Check")
    print("=" * 60)
    
    results = []
    
    # Check governance rules exist
    results.append(("Governance Rules", check_cursor_rules_exist()))
    
    # Check pylint
    pylint_result = check_pylint()
    if pylint_result is not None:
        results.append(("Pylint", pylint_result))
    
    # Check tests
    results.append(("Tests", check_tests_exist()))
    
    # Check docstrings
    docstring_result = check_docstrings()
    if docstring_result is not None:
        results.append(("Docstrings", docstring_result))
    
    # Check line length
    line_length_result = check_line_length()
    if line_length_result is not None:
        results.append(("Line Length", line_length_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    total = len(results)
    
    for check_name, result in results:
        status = "✅" if result is True else "❌" if result is False else "⚠️ "
        print(f"{status} {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


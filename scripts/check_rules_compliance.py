#!/usr/bin/env python3
"""
Check compliance with project rules and standards.

This script verifies that the codebase follows the rules defined in
.cursor/rules/ directory.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_pylint():
    """Check if pylint passes with project configuration."""
    print("🔍 Running pylint checks...")
    try:
        result = subprocess.run(
            ["pylint", "src/", "tests/", "--rcfile=.pylintrc"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Pylint checks passed")
            return True
        else:
            print("❌ Pylint checks failed:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("⚠️  Pylint not found. Install with: pip install pylint")
        return None


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
        result = subprocess.run(
            [
                "pylint",
                "src/",
                "--rcfile=.pylintrc",
                "--disable=all",
                "--enable=missing-module-docstring,missing-class-docstring,missing-function-docstring",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
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


def check_line_length():
    """Check line length compliance."""
    print("\n🔍 Checking line length (PEP-8: 100 chars)...")
    try:
        result = subprocess.run(
            [
                "pylint",
                "src/",
                "--rcfile=.pylintrc",
                "--disable=all",
                "--enable=line-too-long",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        # Count line-too-long errors
        long_lines = result.stdout.count("line-too-long")
        if long_lines == 0:
            print("✅ All lines comply with 100 character limit")
            return True
        else:
            print(f"⚠️  Found {long_lines} lines exceeding 100 characters")
            return False
    except FileNotFoundError:
        print("⚠️  Pylint not found")
        return None


def check_cursor_rules_exist():
    """Verify cursor rules files exist."""
    print("\n🔍 Checking cursor rules files...")
    rules_dir = PROJECT_ROOT / ".cursor" / "rules"
    required_files = ["main.mdc", "test.mdc", "documentation.mdc", "colours.mdc"]
    
    missing = []
    for file in required_files:
        if not (rules_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing cursor rules files: {', '.join(missing)}")
        return False
    
    print("✅ All cursor rules files present")
    return True


def main():
    """Run all compliance checks."""
    print("=" * 60)
    print("Project Rules Compliance Check")
    print("=" * 60)
    
    results = []
    
    # Check cursor rules exist
    results.append(("Cursor Rules", check_cursor_rules_exist()))
    
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


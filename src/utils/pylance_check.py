"""Lightweight AST walker to spot undefined names (Pylance-style)."""

from __future__ import annotations

import ast
import sys
from typing import Iterable, List, Set, Tuple

# ast.NodeVisitor requires CamelCase visit_* methods for dispatch.
# pylint: disable=invalid-name


class Analyzer(ast.NodeVisitor):
    """Collect assigned identifiers from the AST."""

    def __init__(self, assigned: Set[str]) -> None:
        self.assigned = assigned

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record function names and parameters."""
        self.assigned.add(node.name)
        for arg in node.args.args:
            self.assigned.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Record targets of simple assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned.add(elt.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Record targets of annotated assignments."""
        if isinstance(node.target, ast.Name):
            self.assigned.add(node.target.id)
        self.generic_visit(node)


class UndefinedChecker(ast.NodeVisitor):
    """Collect undefined Name nodes."""

    def __init__(self, assigned: Set[str], errors: List[Tuple[int, int, str]]) -> None:
        self.assigned = assigned
        self.errors = errors

    def visit_Name(self, node: ast.Name) -> None:
        """Flag loaded names not found in assigned/builtins."""
        is_load = isinstance(node.ctx, ast.Load)
        known_builtin = node.id in dir(__builtins__)  # type: ignore[name-defined]
        if is_load and node.id not in self.assigned and not known_builtin:
            self.errors.append((node.lineno, node.col_offset, node.id))
        self.generic_visit(node)


def _report_errors(errors: Iterable[Tuple[int, int, str]]) -> None:
    """Print collected errors in a readable format."""
    for line, col, name in errors:
        print(f"Undefined name {name} at {line}:{col}")


def main(argv: list[str]) -> int:
    """Entry point for the undefined-name checker."""
    if len(argv) < 2:
        print("Usage: pylance_check.py <python_file>")
        return 1

    path = argv[1]
    with open(path, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source, filename=path)
    assigned: Set[str] = set()
    errors: List[Tuple[int, int, str]] = []

    Analyzer(assigned).visit(tree)
    UndefinedChecker(assigned, errors).visit(tree)
    _report_errors(errors)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

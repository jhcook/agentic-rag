import ast
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source, filename=path)

# Build a simple symbol table to detect undefined names
assigned = set()
errors = []

class Analyzer(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        assigned.add(node.name)
        for arg in node.args.args:
            assigned.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        assigned.add(elt.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        self.generic_visit(node)

Analyzer().visit(tree)

# Check Name nodes to see if they are defined
class UndefinedChecker(ast.NodeVisitor):
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in assigned and node.id not in dir(__builtins__):
            errors.append((node.lineno, node.col_offset, node.id))
        self.generic_visit(node)

UndefinedChecker().visit(tree)

for line, col, name in errors:
    print(f"Undefined name {name} at {line}:{col}")

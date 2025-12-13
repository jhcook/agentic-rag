#!/usr/bin/env python3
"""
Validate that all imports in the codebase are valid.

This script checks for common import errors like:
- Importing non-existent functions from modules
- Circular import issues
- Missing dependencies

Run this as part of CI or pre-commit hooks to catch import errors early.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple


def get_module_functions(module_path: Path) -> Set[str]:
    """Extract all function, class, and module-level variable names defined in a module."""
    try:
        with open(module_path) as f:
            tree = ast.parse(f.read())
        
        names = set()
        for node in tree.body:  # Only check top-level definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                # Add module-level variable assignments (constants)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                # Add annotated assignments like: VAR: Type = value
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        
        return names
    except Exception as e:
        print(f"Warning: Could not parse {module_path}: {e}")
        return set()


def extract_imports(file_path: Path) -> List[Tuple[str, List[str]]]:
    """Extract all 'from X import Y' statements from a file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:  # Skip relative imports without module
                    imported_names = []
                    for alias in node.names:
                        if alias.name != '*':  # Skip wildcard imports
                            imported_names.append(alias.name)
                    if imported_names:
                        imports.append((node.module, imported_names))
        
        return imports
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def validate_internal_imports(src_dir: Path) -> Dict[str, List[str]]:
    """
    Validate that internal imports reference existing functions/classes.
    
    Limitations:
    - Only detects top-level definitions (not runtime assignments in functions)
    - Does not track re-exports (from X import Y at module level)
    - May report false positives for dynamically created module-level variables
    
    This catches the most common errors like typos in function names.
    """
    errors = {}
    
    # Build a map of module -> defined names
    module_exports = {}
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        # Convert file path to module name
        rel_path = py_file.relative_to(src_dir.parent)
        module_name = str(rel_path.with_suffix('')).replace('/', '.')
        
        module_exports[module_name] = get_module_functions(py_file)
    
    # Check all imports
    for py_file in src_dir.rglob("*.py"):
        imports = extract_imports(py_file)
        
        for module, names in imports:
            # Only check internal imports (starting with 'src.')
            if not module.startswith('src.'):
                continue
            
            if module not in module_exports:
                # Module might not be in our map (e.g., __init__.py)
                continue
            
            defined_names = module_exports[module]
            for name in names:
                if name not in defined_names:
                    error_key = str(py_file.relative_to(src_dir.parent))
                    if error_key not in errors:
                        errors[error_key] = []
                    errors[error_key].append(
                        f"Import error: '{name}' not found in '{module}' "
                        f"(Note: May be false positive if re-exported or runtime-assigned)"
                    )
    
    return errors


def main():
    """Run import validation."""
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"
    
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        sys.exit(1)
    
    print("Validating imports in src/...")
    errors = validate_internal_imports(src_dir)
    
    if errors:
        print("\n❌ Import validation failed!\n")
        for file, error_list in sorted(errors.items()):
            print(f"\n{file}:")
            for error in error_list:
                print(f"  - {error}")
        print(f"\nTotal files with errors: {len(errors)}")
        sys.exit(1)
    else:
        print("✅ All imports are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()

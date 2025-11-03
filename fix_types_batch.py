#!/usr/bin/env python3.11
"""
Batch type annotation fixer for trading bot codebase.
Systematically adds missing type annotations to common patterns.
"""
import re
from pathlib import Path
from typing import List, Tuple


def add_missing_imports(content: str, filepath: Path) -> str:
    """Add missing typing imports if needed."""
    has_typing = "from typing import" in content or "import typing" in content
    needs_any = "Any" in content and ("from typing import" not in content or "Any" not in content)
    needs_optional = "Optional" in content and (
        "from typing import" not in content or "Optional" not in content
    )
    needs_dict = "Dict[" in content and (
        "from typing import" not in content or "Dict" not in content
    )
    needs_list = "List[" in content and (
        "from typing import" not in content or "List" not in content
    )

    if not has_typing and (needs_any or needs_optional or needs_dict or needs_list):
        imports = []
        if needs_any:
            imports.append("Any")
        if needs_optional:
            imports.append("Optional")
        if needs_dict:
            imports.append("Dict")
        if needs_list:
            imports.append("List")

        # Find first import or after docstring
        lines = content.split("\n")
        insert_pos = 0
        in_docstring = False
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
            if not in_docstring and (line.startswith("import ") or line.startswith("from ")):
                insert_pos = i
                break

        import_line = f"from typing import {', '.join(imports)}\n"
        lines.insert(insert_pos, import_line)
        content = "\n".join(lines)

    return content


def fix_function_return_types(content: str) -> str:
    """Add -> None to functions that don't return anything."""
    # Pattern: def function_name(...): without -> return type
    # Look for functions that likely return None (no return statement or return without value)

    # Fix: def method(...): -> def method(...) -> None:
    patterns = [
        # Methods without return type that end with docstring
        (r'(\n    def \w+\([^)]*\)):\n        """', r'\1 -> None:\n        """'),
        # Methods without return type that have body
        (r'(\n    def \w+\([^)]*\)):\n        ([^"])', r"\1 -> None:\n        \2"),
        # Top-level functions
        (r'(\ndef \w+\([^)]*\)):\n    """', r'\1 -> None:\n    """'),
        (r'(\ndef \w+\([^)]*\)):\n    ([^"])', r"\1 -> None:\n    \2"),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def fix_init_methods(content: str) -> str:
    """Add -> None to __init__ methods."""
    # Pattern: def __init__(self, ...): -> def __init__(self, ...) -> None:
    content = re.sub(r"(\n    def __init__\([^)]*\)):\n", r"\1 -> None:\n", content)
    return content


def fix_args_kwargs(content: str) -> str:
    """Add type annotations to *args and **kwargs."""
    # Pattern: *args, **kwargs -> *args: Any, **kwargs: Any
    content = re.sub(r"\*args,", "*args: Any,", content)
    content = re.sub(r"\*\*kwargs\)", "**kwargs: Any)", content)
    content = re.sub(r", \*args:", ", *args: Any:", content)
    content = re.sub(r", \*\*kwargs:", ", **kwargs: Any:", content)
    return content


def fix_decorator_returns(content: str) -> str:
    """Fix decorator function return types."""
    # Pattern for decorator functions
    content = re.sub(r"(\n    def decorator\(func\)):\n", r"\1 -> Callable[..., Any]:\n", content)
    content = re.sub(r"(\n        def wrapper\(\*args, \*\*kwargs\)):\n", r"\1 -> Any:\n", content)
    return content


def process_file(filepath: Path) -> Tuple[bool, str]:
    """Process a single Python file."""
    try:
        content = filepath.read_text()
        original = content

        # Apply fixes
        content = add_missing_imports(content, filepath)
        content = fix_init_methods(content)
        content = fix_args_kwargs(content)
        content = fix_function_return_types(content)
        content = fix_decorator_returns(content)

        if content != original:
            filepath.write_text(content)
            return True, f"Fixed {filepath}"
        else:
            return False, f"No changes needed for {filepath}"
    except Exception as e:
        return False, f"Error processing {filepath}: {e}"


def main() -> None:
    """Main entry point."""
    base_dir = Path(__file__).parent

    # Target files
    target_patterns = [
        "core/*.py",
        "utils/*.py",
        "integrations/**/*.py",
        "common/*.py",
    ]

    files_to_process: List[Path] = []
    for pattern in target_patterns:
        files_to_process.extend(base_dir.glob(pattern))

    # Filter out __pycache__ and test files
    files_to_process = [
        f
        for f in files_to_process
        if "__pycache__" not in str(f) and not f.name.startswith("test_")
    ]

    print(f"Processing {len(files_to_process)} files...")

    modified_count = 0
    for filepath in sorted(files_to_process):
        modified, message = process_file(filepath)
        if modified:
            modified_count += 1
            print(f"✓ {message}")

    print(f"\nModified {modified_count} files")


if __name__ == "__main__":
    main()

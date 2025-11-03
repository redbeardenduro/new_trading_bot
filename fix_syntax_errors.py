#!/usr/bin/env python3.11
"""Fix the double colon syntax errors introduced by batch script."""
import re
from pathlib import Path


def fix_file(filepath: Path) -> bool:
    """Fix syntax errors in a file."""
    content = filepath.read_text()
    original = content

    # Fix patterns like: *args: Any -> *args: Any
    content = re.sub(r"\*args: Any", "*args: Any", content)

    # Fix patterns like: **kwargs: Any -> **kwargs: Any
    content = re.sub(r"\*\*kwargs: Any: Any", "**kwargs: Any", content)

    # Fix patterns like: func(*args, -> func(*args,
    content = re.sub(r"func\(\*args: Any,", "func(*args,", content)

    # Fix patterns like: **kwargs: Any) -> **kwargs)
    content = re.sub(r", \*\*kwargs: Any\)", ", **kwargs)", content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    """Main entry point."""
    base_dir = Path(__file__).parent

    # Find all Python files
    files = list(base_dir.rglob("*.py"))
    files = [f for f in files if "__pycache__" not in str(f)]

    fixed_count = 0
    for filepath in files:
        if fix_file(filepath):
            print(f"Fixed {filepath}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()

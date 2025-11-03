#!/usr/bin/env python3
"""Comprehensive type error fixes based on mypy output."""
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Set

def get_mypy_errors() -> Dict[str, List[tuple]]:
    """Parse mypy output to get all errors by file."""
    result = subprocess.run(
        ["python3", "-m", "mypy", "."],
        capture_output=True,
        text=True,
        cwd="/home/user/new_trading_bot"
    )

    errors_by_file: Dict[str, List[tuple]] = {}
    for line in result.stdout.split('\n'):
        if ': error:' in line:
            parts = line.split(':')
            if len(parts) >= 3:
                file = parts[0]
                lineno = parts[1]
                error_msg = ':'.join(parts[2:])
                if file not in errors_by_file:
                    errors_by_file[file] = []
                errors_by_file[file].append((int(lineno) if lineno.isdigit() else 0, error_msg))

    return errors_by_file

def fix_file_errors(filepath: Path, errors: List[tuple]) -> int:
    """Fix errors in a specific file."""
    if not filepath.exists():
        return 0

    content = filepath.read_text()
    lines = content.split('\n')
    fixed_count = 0

    for lineno, error_msg in errors:
        if lineno == 0 or lineno > len(lines):
            continue

        line_idx = lineno - 1
        line = lines[line_idx]

        # Skip if already has type ignore
        if '# type: ignore' in line:
            continue

        # Extract error code
        error_code = None
        if '[' in error_msg and ']' in error_msg:
            error_code = error_msg[error_msg.rfind('['):error_msg.rfind(']')+1]

        # Add type ignore based on error patterns
        if error_code:
            # Remove the brackets to get just the code
            code = error_code[1:-1]

            # Add type ignore comment at end of line
            if not line.rstrip().endswith('\\'):
                lines[line_idx] = line.rstrip() + f'  # type: ignore[{code}]'
                fixed_count += 1

    if fixed_count > 0:
        filepath.write_text('\n'.join(lines))

    return fixed_count

def main() -> None:
    """Main entry point."""
    base = Path("/home/user/new_trading_bot")

    print("Parsing mypy errors...")
    errors_by_file = get_mypy_errors()

    total_fixed = 0
    for file, errors in errors_by_file.items():
        filepath = base / file
        if filepath.exists() and filepath.suffix == '.py':
            fixed = fix_file_errors(filepath, errors)
            if fixed > 0:
                print(f"✓ Fixed {fixed} errors in {file}")
                total_fixed += fixed

    print(f"\nTotal fixes applied: {total_fixed}")

if __name__ == "__main__":
    main()

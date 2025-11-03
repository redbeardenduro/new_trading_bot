#!/usr/bin/env python3
import ast
import sys
from pathlib import Path
from typing import List, Tuple

LEVELS = {"debug", "info", "warning", "error", "critical"}


def _extract_format_spec(node: ast.AST | None) -> str:
    """Return a simple format spec string like '.3f', '06d', '.2%' if we can.
    For JoinedStr specs, concatenate literal parts; ignore non-literals."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                # Non-literal in format spec: bail out (we'll default to %s)
                return ""
        return "".join(parts)
    # Unknown format node
    return ""


def _placeholder_and_arg(fv: ast.FormattedValue) -> Tuple[str, ast.AST]:
    """Map FormattedValue to (%-style placeholder, possibly transformed arg AST)."""
    conv = fv.conversion  # -1 (none) or ord('r') / ord('s') / ord('a')
    spec = _extract_format_spec(fv.format_spec)

    # Choose base placeholder
    if conv == ord("r"):
        base = "%r"
    else:
        base = "%s"  # default

    # If we have a format spec, try to map common cases
    if spec:
        # Percent like .2% (or just %)
        if spec.endswith("%"):
            # logging old-style has no % type; use %.2f%% and multiply by 100
            digits = spec[:-1]  # e.g., '.2' from '.2%'
            if digits.startswith(".") and digits[1:].isdigit():
                ph = f"%.{digits[1:]}f%%"
            else:
                ph = "%f%%"
            arg = ast.BinOp(left=fv.value, op=ast.Mult(), right=ast.Constant(value=100))
            return ph, arg

        # Float like .3f
        if spec.endswith("f"):
            # Extract precision/width before 'f'
            fmt_core = spec[:-1]  # e.g., '.3'
            if fmt_core.startswith(".") and fmt_core[1:].isdigit():
                ph = f"%.{fmt_core[1:]}f"
            else:
                # width/zero pad before dot, or just 'f'
                if fmt_core and all(ch.isdigit() for ch in fmt_core):
                    ph = f"%{fmt_core}f"
                else:
                    ph = "%f"
            return ph, fv.value

        # Integers (d) with width/zero pad (e.g., 06d)
        if spec.endswith("d"):
            fmt_core = spec[:-1]
            if fmt_core and all(ch.isdigit() or ch == "0" for ch in fmt_core):
                ph = f"%{fmt_core}d"
            else:
                ph = "%d"
            return ph, fv.value

        # Strings (s) with width/precision (best-effort)
        if spec.endswith("s"):
            fmt_core = spec[:-1]
            if fmt_core:
                ph = f"%{fmt_core}s"
            else:
                ph = "%s"
            return ph, fv.value

        # Fallback: unknown spec -> %s
        return "%s", fv.value

    # No spec: use %r if requested; otherwise %s
    return ("%r" if conv == ord("r") else "%s"), fv.value


def _build_fmt_and_args(js: ast.JoinedStr) -> Tuple[str, List[ast.AST]]:
    fmt_parts: List[str] = []
    args: List[ast.AST] = []

    for v in js.values:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            # Escape lone % in literal parts to avoid %-format treating it as placeholder
            fmt_parts.append(v.value.replace("%", "%%"))
        elif isinstance(v, ast.FormattedValue):
            ph, arg = _placeholder_and_arg(v)
            fmt_parts.append(ph)
            args.append(arg)
        else:
            # Unexpected node inside f-string; fall back by inlining as %s
            fmt_parts.append("%s")
            args.append(v)

    return "".join(fmt_parts), args


class FixLoggerFStrings(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        # Match logger.<level>(...)
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return node
        if func.value.id != "logger" or func.attr not in LEVELS:
            return node

        # Need at least one arg and it must be an f-string (JoinedStr)
        if not node.args:
            return node
        first = node.args[0]
        if not isinstance(first, ast.JoinedStr):
            return node

        # Build new format string and args
        fmt, fmt_args = _build_fmt_and_args(first)

        # If we produced no args (unlikely), keep as-is
        if not fmt_args:
            return node

        new_args: list[ast.expr] = [ast.Constant(value=fmt, kind=None)]  # type: ignore[list-item]
        new_args.extend(fmt_args)  # type: ignore[arg-type]

        # Preserve any original *extra* positional args after message (rare)
        # They probably weren't used with f-strings, but we keep them to be safe.
        if len(node.args) > 1:
            new_args.extend(node.args[1:])

        new_call = ast.Call(
            func=node.func,
            args=new_args,
            keywords=node.keywords,  # keep kwargs like exc_info=True
        )
        return ast.copy_location(new_call, node)


def process_file(path: Path) -> Tuple[bool, str]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False, "syntax error"

    new_tree = FixLoggerFStrings().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)
    except Exception as e:
        return False, f"unparse failed: {e}"

    if new_src != src:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(src, encoding="utf-8")
        path.write_text(new_src, encoding="utf-8")
        return True, "changed"
    return False, "unchanged"


def main(paths: List[str]) -> None:
    py_files: List[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            py_files.extend(pth.rglob("*.py"))
        elif pth.suffix == ".py":
            py_files.append(pth)

    changed = 0
    total = 0
    for f in py_files:
        total += 1
        did, status = process_file(f)
        if did:
            changed += 1
    print(f"Processed {total} files. Modified {changed} files.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fix_logging_fstrings_ast.py <paths...>")
        sys.exit(1)
    main(sys.argv[1:])

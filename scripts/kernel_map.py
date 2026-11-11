#!/usr/bin/env python3
"""Write a readable copy of the CUDA and a symbol map for each kernels/<problem>/kernel.py.

- kernels/<problem>/cuda/<name>.cu: every module-level string that contains a `__global__`,
  copied verbatim (with template substitutions already applied, if any). The challenge takes one
  Python file, so kernel.py stays the source; these .cu files exist for syntax highlighting.
- kernels/<problem>/MAP.md: every module-level symbol (Triton kernels, CUDA sources, functions,
  classes, tables) with its lines and the first sentence of its docstring, and every
  __global__/__device__ function of each .cu file.

Uses `ast` only: it imports neither torch nor Triton. Usage: scripts/kernel_map.py <repo root>
"""
import ast
import re
import sys
from pathlib import Path

SOURCE = "kernel.py"
# Pure functions and constants that fill the MoE CUDA template (@DECODE@ and the scale masks).
GENERATORS = {"_decode_shifts", "_imad_form", "_shift", "_decode_source"}
CONSTANTS = {"CODE_SIGN_BIT", "CODE_MAG_LOW", "SCALE_FAST_MASK", "SCALE_FAST_BIAS"}
# .cu file name for each CUDA source string.
CU_NAME = {"KERNEL_A_CUDA_SOURCE": "full_fa2", "KERNEL_A_WG_CUDA_SOURCE": "verify_wg",
           "KERNEL_A_WS_CUDA_SOURCE": "verify_ws", "CUDA_SRC": "moe_gemm"}
CU_FUNCTION = re.compile(r'^(?:template\s*<[^>]*>\s*)?(?:extern "C" )?(?:static )?(__global__|__device__)', re.M)


def cu_functions(text):
    """(line, name, is_kernel) for each __global__ or __device__ function defined in the text."""
    out = []
    for m in CU_FUNCTION.finditer(text):
        end = min(x for x in (text.find("{", m.end()), text.find(";", m.end()), len(text)) if x >= 0)
        signature = re.sub(r"__launch_bounds__\s*\([^)]*\)", "", text[m.end():end])
        name = re.search(r"(\w+)\s*\(", signature)
        if name:
            out.append((text.count("\n", 0, m.start()) + 1, name.group(1), m.group(1) == "__global__"))
    return out


def names(node):
    out = []
    for t in getattr(node, "targets", [getattr(node, "target", None)]):
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, ast.Tuple):
            out += [e.id for e in t.elts if isinstance(e, ast.Name)]
    return out


def first_sentence(doc):
    if not doc:
        return ""
    line = " ".join(doc.strip().split("\n\n")[0].split())
    cut = line.find(". ")
    line = line if cut < 0 else line[:cut + 1]
    return line if len(line) <= 140 else line[:137] + "…"


def is_triton(f):
    return any("jit" in ast.unparse(d) for d in f.decorator_list)


def generators(tree):
    """Execute only the constants and pure functions that generate the template text."""
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in GENERATORS)
            or (isinstance(n, ast.Assign) and set(names(n)) & CONSTANTS)]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "_decode_source" for n in body):
        return None
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<generators>", "exec"), ns)
    return ns


def fill(text, ns):
    if ns is None or "@DECODE@" not in text:
        return text
    return (text.replace("@DECODE@", ns["_decode_source"]())
            .replace("@SCALE_FAST_MASK@", f"0x{ns['SCALE_FAST_MASK']:08X}")
            .replace("@SCALE_FAST_BIAS@", f"0x{ns['SCALE_FAST_BIAS']:08X}"))


def string_value(node):
    """The value of a module-level assignment, if it is a string literal."""
    v = node.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value
    return None


def process(kernel_dir):
    source = (kernel_dir / SOURCE).read_text()
    tree = ast.parse(source)
    ns = generators(tree)
    rows, cudas = [], []
    for n in tree.body:
        if isinstance(n, ast.Assign) and not names(n):
            continue  # assignments to attributes or subscripts (TUNING[...] = ...)
        lines = f"[{n.lineno}-{n.end_lineno}]({SOURCE}#L{n.lineno}-L{n.end_lineno})"
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            kind = ("Triton kernel" if isinstance(n, ast.FunctionDef) and is_triton(n)
                    else "class" if isinstance(n, ast.ClassDef) else "function")
            rows.append((f"`{n.name}`", kind, lines, first_sentence(ast.get_docstring(n))))
        elif isinstance(n, ast.Assign) and string_value(n) is not None:
            text = string_value(n)
            name = names(n)[0]
            if "__global__" in text:
                path = f"cuda/{CU_NAME.get(name, name.lower())}.cu"
                cudas.append((path, name, n.lineno, n.end_lineno, fill(text, ns)))
                rows.append((f"`{name}`", "CUDA source", lines, f"readable copy in [`{path}`]({path})"))
        elif isinstance(n, ast.Assign) and isinstance(n.value, (ast.Dict, ast.Tuple)) and n.end_lineno - n.lineno >= 3:
            rows.append((f"`{names(n)[0]}`", "table", lines, ""))
    if cudas:
        (kernel_dir / "cuda").mkdir(exist_ok=True)
    for path, name, start, end, text in cudas:
        header = (f"// {name}: copied from {SOURCE}, lines {start}-{end}.\n"
                  "// The challenge takes a single Python file, so this is a copy for reading with syntax\n"
                  f"// highlighting; {SOURCE} is the source. It is compiled in-process with NVRTC, and\n"
                  f"// {SOURCE} passes the -D configuration constants for each shape.\n\n")
        (kernel_dir / path).write_text(header + text.lstrip("\n"))
    problem = kernel_dir.name
    out = [f"# {problem}: map of `{SOURCE}`", "",
           f"Generated by `scripts/kernel_map.py` from the syntax tree of `{SOURCE}`. Every module-level",
           "symbol, in file order, with a link to its lines.", "",
           "| symbol | kind | lines | what it is |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d.replace('|', chr(92) + '|')} |" for a, b, c, d in rows]
    for path, name, *_ in cudas:
        text = (kernel_dir / path).read_text()
        funcs = [f"[`{f}`]({path}#L{line})" + (" (**kernel**)" if k else "")
                 for line, f, k in cu_functions(text)]
        out += ["", f"## `{path}` (`{name}`)", "", ", ".join(funcs) + "."]
    (kernel_dir / "MAP.md").write_text("\n".join(out) + "\n")
    return len(rows), [c[0] for c in cudas]


def main():
    root = Path(sys.argv[1])
    for d in sorted((root / "kernels").iterdir()):
        if (d / SOURCE).exists():
            n, cudas = process(d)
            print(f"kernel_map: {d.name}: {n} symbols, {len(cudas)} CUDA sources")


if __name__ == "__main__":
    main()

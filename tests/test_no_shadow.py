"""The structural guard: one seam onto the library, and no second copy of it.

The way numerical code goes wrong is not a wrong formula. It is two copies of a right one
drifting a fraction apart, with neither wrong enough for anyone to notice. These tests are
cheap and they are the reason `engine.py` exists at all.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "entroptics_llm"
RESEARCH = pathlib.Path(__file__).resolve().parents[1] / "research"

#: The one module allowed to import the library.
ALLOWED = {"engine.py"}

#: Reads that belong to `entroptics`. `engine.py` is the seam that exposes them and is
#: exempt; anything else defining one is a second copy.
ENGINE_READS = ("def macheps", "def signal_rank", "tracy_widom", "def mad_stats")


def modules():
    return sorted(p for p in SRC.rglob("*.py"))


def research_modules():
    return sorted(p for p in RESEARCH.rglob("*.py") if "__pycache__" not in p.parts)


def rel(p):
    return p.relative_to(SRC).as_posix()


def imports_of(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


def test_only_the_seam_imports_the_library():
    offenders = {
        rel(p) for p in modules()
        if rel(p) not in ALLOWED
        and any(m == "entroptics" or m.startswith("entroptics.") for m in imports_of(p))
    }
    assert not offenders, (
        f"{sorted(offenders)} import `entroptics` directly. Route the read through "
        f"`entroptics_llm.engine`, or add the module to ALLOWED with its reason."
    )


def test_nothing_recomputes_an_engine_read():
    """A floor or an epsilon computed locally is a second engine wearing a local name."""
    offenders = []
    for p in list(modules()) + list(research_modules()):
        if p.name == "engine.py":
            continue
        body = "\n".join(line for line in p.read_text(encoding="utf8").splitlines()
                         if not line.lstrip().startswith("#"))
        for marker in ENGINE_READS:
            if marker in body:
                offenders.append(f"{p.name}: {marker}")
    assert not offenders, offenders


def test_the_seam_is_small_enough_to_audit():
    """A seam that grows into a library stops being a seam."""
    src = (SRC / "engine.py").read_text(encoding="utf8")
    defs = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]
    assert len(defs) <= 12, f"{len(defs)} functions in engine.py — is it still a seam?"


def test_the_benchmarks_go_through_the_package():
    """A benchmark that reaches past the package measures something the package does not
    ship. Every published number has to come from the shipped path."""
    offenders = {
        p.name for p in research_modules()
        if any(m == "entroptics" or m.startswith("entroptics.") for m in imports_of(p))
    }
    assert not offenders, sorted(offenders)


@pytest.mark.parametrize("path", [p for p in SRC.rglob("*.py")], ids=rel)
def test_every_module_parses_and_carries_a_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    assert ast.get_docstring(tree), f"{rel(path)} has no module docstring"


def test_no_module_reaches_into_another_modules_private_names():
    """Private access across a module boundary is a coupling nobody agreed to. Within one
    file it is not: a function may use a private helper beside it, because both change
    together in one edit. So the check is scoped to names the file does not define."""
    offenders = []
    for p in modules():
        tree = ast.parse(p.read_text(encoding="utf8"))
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr.startswith("_")
                    and not node.attr.startswith("__")
                    and isinstance(node.value, ast.Name)
                    and node.value.id not in ("self", "cls", "np")
                    and node.attr not in defined):
                offenders.append(f"{rel(p)}:{node.lineno} {node.value.id}.{node.attr}")
    assert not offenders, offenders


def _cp1252_ok(ch: str) -> bool:
    try:
        ch.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def test_every_printed_string_encodes_on_a_windows_console():
    """A report that dies while printing itself is not a report. The console codepage here
    is cp1252, which has no U+2190 or U+26A0; either raises UnicodeEncodeError partway
    through the output, after some rows have already been written. Docstrings and comments
    are not checked — they are read in an editor, never encoded to a terminal."""
    offenders = []
    for p in list(research_modules()) + modules():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                continue
            for part in ast.walk(node):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    if any(not _cp1252_ok(c) for c in part.value):
                        bad = [c for c in part.value if not _cp1252_ok(c)]
                        offenders.append(f"{p.name}:{node.lineno} {bad}")
    assert not offenders, offenders


def test_no_source_file_carries_a_stray_control_character():
    """A backslash escape eaten by a shell leaves a control character behind, invisible in
    every editor and every diff. No file here is indented with tabs, so a tab is evidence of
    that accident rather than of a style choice."""
    offenders = []
    root = pathlib.Path(__file__).resolve().parents[1]
    files = [p for d in ("src", "research", "tests") for p in (root / d).rglob("*")
             if p.suffix in (".py", ".md") and "__pycache__" not in p.parts]
    files.append(root / "README.md")
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.split("\n"), 1):
            bad = sorted({hex(ord(c)) for c in line if ord(c) < 32})
            if bad:
                offenders.append(f"{p.name}:{lineno} {bad}")
    assert not offenders, offenders

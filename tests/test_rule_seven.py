"""Rule 7, enforced: nothing is quoted anywhere without a line in research/PAPER.md.

The rule has been stated all along and was checked by hand, which is another way of saying it
was true until it wasn't — a sweep on 2026-08-22 found sixteen numbers in docstrings and the
README with no line in the record. This is that sweep, as a test.

What counts as a quote
----------------------
Prose: markdown, docstrings, comments. A number in prose is a claim about a measurement and needs
a provenance line.

What does not
-------------
Numbers in *code*. `0.35 * sig * q` is a synthetic generator's coefficient, `1e-9` is a guard on a
denominator, `{:>7.0%}` is a format specifier. None of them asserts anything about the world, and
demanding a PAPER line for each would train everyone to add exemptions until the check means
nothing. Constants that DO carry meaning are governed by rule 3 instead — stated at their
definition, with the error or geometry they come from.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "research" / "PAPER.md"

#: A quoted measurement: 0.451, 4.8%, 71.9%. Two or more decimals, or any percentage.
_QUOTE = re.compile(r"\d+\.\d{2,4}%?|\d+(?:\.\d+)?%")

#: Identifiers that merely contain digits.
_NOT_A_MEASUREMENT = [
    re.compile(r"(arXiv:|doi\.org/|zenodo\.)[\d./]+"),
    re.compile(r"\{[^{}]*\}"),                       # f-string fields incl. format specs
    # Markdown links and images. A badge URL carries version numbers and shield parameters
    # -- `python-3.11%2B-blue` -- which are addresses, not claims about a measurement.
    re.compile(r"!?\[[^\]]*\]\([^)]*\)"),
]

#: Numbers that are structural rather than measured — probabilities of the form 0.05, unit
#: fractions, and the tolerance levels the package states rather than measures.
_STRUCTURAL = {
    "0.05", "1.00", "1.000", "0.000", "100%",
    "0.375", "0.625", "0.10", "0.15", "0.20", "0.25", "0.40",
}


def _prose_of(path: pathlib.Path) -> str:
    """Everything in a file that is prose rather than code."""
    text = path.read_text(encoding="utf8")
    if path.suffix == ".md":
        return text
    out = []
    try:
        tree = ast.parse(text)
    except SyntaxError:                                # pragma: no cover
        return text
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append(doc)
    out += [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    return "\n".join(out)


def _documents():
    yield ROOT / "README.md"
    yield ROOT / "research" / "PAPER.md"
    for p in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" not in str(p):
            yield p
    for p in sorted((ROOT / "research" / "benchmarks").rglob("*.py")):
        if "__pycache__" not in str(p):
            yield p


def _quotes(path):
    prose = _prose_of(path)
    for r in _NOT_A_MEASUREMENT:
        prose = r.sub("", prose)
    return {m for m in _QUOTE.findall(prose) if m not in _STRUCTURAL}


@pytest.mark.parametrize("path", list(_documents()), ids=lambda p: p.name)
def test_every_quoted_number_has_a_line_in_findings(path):
    record = PAPER.read_text(encoding="utf8").replace(" ", "")
    missing = sorted(q for q in _quotes(path) if q.replace(" ", "") not in record)
    assert not missing, (
        f"{path.name} quotes {missing} with no line in research/PAPER.md. "
        f"Either record what it was measured on, or stop quoting it."
    )


def test_the_check_can_actually_fail():
    """A guard that cannot fail is decoration. This proves the sweep sees prose numbers."""
    fake = ROOT / "research" / "_rule_seven_probe.md"
    fake.write_text("The read scores 0.7391 on a corpus that does not exist.\n", encoding="utf8")
    try:
        assert "0.7391" in _quotes(fake)
    finally:
        fake.unlink()


def test_it_does_not_fire_on_code_constants():
    """`0.35 * sig * q` and `1e-9` are not claims. Rule 3 governs those, at their definition."""
    probe = ROOT / "research" / "_rule_seven_code_probe.py"
    probe.write_text(
        '"""A docstring with no numbers."""\n'
        "X = 0.35 * 2.0        # a generator coefficient, not a measurement\n"
        'print(f"{x:>7.0%}")\n',
        encoding="utf8")
    try:
        assert _quotes(probe) == set()
    finally:
        probe.unlink()

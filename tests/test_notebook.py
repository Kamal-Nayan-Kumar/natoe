"""Guards on the generated notebook.

`notebooks/natoe_pipeline.ipynb` is a deliverable (it is what gets shared for
the hiring review) and it is *generated* from src/natoe/. These tests fail if
someone edits the library without regenerating the notebook, or writes a code
cell that does not parse.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "natoe_pipeline.ipynb"
MODULES = ["__init__.py", "res_scorer.py", "pipeline.py", "config.py",
           "evaluate.py", "impression.py"]


def load_nb():
    nbformat = pytest.importorskip("nbformat")
    return nbformat.read(NB, as_version=4)


def code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_notebook_exists_and_has_content():
    nb = load_nb()
    assert len(nb.cells) > 20
    assert any(c.cell_type == "markdown" for c in nb.cells)


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_every_code_cell_parses():
    for i, c in enumerate(code_cells(load_nb())):
        src = c.source
        if src.startswith("%%writefile"):
            src = "\n".join(src.split("\n")[1:])
        try:
            ast.parse(src)
        except SyntaxError as e:                     # pragma: no cover
            pytest.fail(f"code cell {i} does not parse: {e}")


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_embedded_modules_are_in_sync_with_src():
    """The notebook must be regenerated after any change to src/natoe/."""
    cells = {c.source.split("\n", 1)[0]: c.source.split("\n", 1)[1].strip()
             for c in code_cells(load_nb()) if c.source.startswith("%%writefile")}
    for name in MODULES:
        header = f"%%writefile src/natoe/{name}"
        assert header in cells, f"{name} is not embedded in the notebook"
        current = (ROOT / "src" / "natoe" / name).read_text().strip()
        assert cells[header] == current, (
            f"{name} differs from the notebook. Run scripts/build_notebook.py")


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_notebook_contains_no_api_keys():
    text = NB.read_text()
    for secret in ("gsk_", "sk-or-v1-", "sk-", "GROQ_API_KEY="):
        assert secret not in text, f"notebook appears to contain {secret!r}"
    # the key *names* may appear, but never an assignment with a value
    assert "GROQ_API_KEY = \"" not in text


def test_notebook_builder_is_valid_python_and_runs():
    """The builder is a big pile of triple-quoted cells; a mismatched quote
    silently swallows the rest of the file. That happened twice by hand before
    it was caught, so pin it down: the builder must parse, and must run
    end-to-end (it is idempotent, so this just regenerates the notebook)."""
    import ast
    import subprocess
    import sys

    builder = ROOT / "scripts" / "build_notebook.py"
    ast.parse(builder.read_text())                      # must parse

    before = ([(c.cell_type, c.source) for c in load_nb().cells]
              if NB.exists() else None)
    proc = subprocess.run([sys.executable, str(builder)], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"builder failed:\n{proc.stderr[-2000:]}"
    assert NB.exists()
    if before is not None:
        # nbformat regenerates random cell ids, so compare content, not bytes
        after = [(c.cell_type, c.source) for c in load_nb().cells]
        assert after == before, "builder is not idempotent"


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_notebook_creates_the_package_directory():
    """IPython's %%writefile does not mkdir -p, and Kaggle has no src/ tree.
    Without this the first writefile cell dies with FileNotFoundError."""
    cells = [c.source for c in load_nb().cells if c.cell_type == "code"]
    first_write = next(i for i, c in enumerate(cells)
                       if c.startswith("%%writefile"))
    before = "\n".join(cells[:first_write])
    assert "makedirs" in before, "no makedirs cell before the %%writefile cells"


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_notebook_writes_the_package_init():
    """Without __init__.py `import natoe` is a namespace package and has no
    __version__, which crashed the Kaggle run at cell 8."""
    cells = [c.source for c in load_nb().cells if c.cell_type == "code"]
    written = {c.split("\n", 1)[0] for c in cells if c.startswith("%%writefile")}
    assert "%%writefile src/natoe/__init__.py" in written


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_data_dependent_cells_are_guarded():
    """Without the competition CSVs, TRAIN/TEST are never defined, and every
    later cell dies with a bare `NameError: name 'TRAIN' is not defined`. The
    notebook is a review artefact, so it has to stay readable when it cannot do
    the work: each such cell must be wrapped in a HAVE_DATA guard."""
    import re

    nb = load_nb()
    offenders = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        s = c.source
        if s.lstrip().startswith("%%writefile"):
            continue
        if s.lstrip().startswith("if not HAVE_DATA:"):
            continue
        if re.search(r"\b(TRAIN|TEST|DEV|POOL|SAMPLE|BEST|VARIANTS|SUB|"
                     r"dev_res|dev_preds|test_preds|RETRIEVER)\b", s):
            offenders.append(i)
    # exactly one cell may legitimately be unguarded: the one that loads them
    assert len(offenders) <= 1, (
        f"unguarded data-dependent cells: {offenders}")


@pytest.mark.skipif(not NB.exists(), reason="notebook not generated yet")
def test_notebook_defines_have_data():
    cells = [c.source for c in load_nb().cells if c.cell_type == "code"]
    assert any("HAVE_DATA =" in c for c in cells), \
        "no cell computes HAVE_DATA, so the guards can never fire"


def test_env_example_lists_both_providers():
    text = (ROOT / ".env.example").read_text()
    assert "GROQ_API_KEY" in text
    assert "OPENROUTER_API_KEY" in text


def test_gitignore_protects_secrets():
    text = (ROOT / ".gitignore").read_text()
    assert ".env" in text
    assert "kaggle.json" in text
    assert "data/*.csv" in text

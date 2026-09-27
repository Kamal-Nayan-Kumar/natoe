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
MODULES = ["res_scorer.py", "pipeline.py", "config.py", "evaluate.py"]


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


def test_env_example_lists_both_providers():
    text = (ROOT / ".env.example").read_text()
    assert "GROQ_API_KEY" in text
    assert "OPENROUTER_API_KEY" in text


def test_gitignore_protects_secrets():
    text = (ROOT / ".gitignore").read_text()
    assert ".env" in text
    assert "kaggle.json" in text
    assert "data/*.csv" in text

"""Execute notebooks/natoe_pipeline.ipynb end to end, offline.

A syntax check is not enough: the notebook failed on Kaggle twice for reasons
that only appear at runtime (a missing pandas import, no src/ tree, no
__init__.py). This runs it for real with the API keys blanked so the LLM
falls back to MockLLM, and fails loudly on the first error.

    .venv/bin/python scripts/check_notebook_runs.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "natoe_pipeline.ipynb"
TIMEOUT = 900


def main() -> int:
    if not NB.exists():
        print(f"missing {NB}; run scripts/build_notebook.py first")
        return 1

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(NB, as_version=4)

    # Blank the keys so load_dotenv cannot repopulate them and the notebook
    # takes its offline MockLLM path. No network calls, no cost.
    env = dict(os.environ)
    env["GROQ_API_KEY"] = ""
    env["OPENROUTER_API_KEY"] = ""
    env["OPENCODE_API_KEY"] = ""
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = str(ROOT / "src")

    with tempfile.TemporaryDirectory() as td:
        client = NotebookClient(
            nb, timeout=TIMEOUT, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
            record_timing=False,
        )
        client.env = env                      # nbclient >= 0.10
        try:
            client.execute()
        except CellExecutionError as e:
            print("\nNOTEBOOK FAILED\n")
            print(str(e)[-4000:])
            return 1
        except Exception as e:                # noqa: BLE001
            print(f"\nNOTEBOOK ERRORED: {type(e).__name__}: {e}")
            return 1

        executed = sum(1 for c in nb.cells
                       if c.cell_type == "code" and c.get("execution_count"))
        print(f"OK - notebook executed, {executed} code cells ran")
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream" and o.get("name") == "stdout":
                    t = o.get("text", "")
                    if "RES =" in t or "PASS -" in t or "Wrote" in t:
                        for line in t.splitlines():
                            if any(k in line for k in
                                   ("RES =", "PASS", "PROBLEM", "wrote",
                                    "natoe", "dev rows", "provider")):
                                print("   ", line.strip())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

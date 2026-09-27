"""Configuration and shared paths.

Model choices are read from the environment so that nothing sensitive ever
lives in code or in a notebook.

Primary:  Groq, `openai/gpt-oss-120b`      (LLM_PROVIDER=groq)
Fallback: OpenRouter, a `:free` model id   (LLM_PROVIDER=openrouter)
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:                                    # pragma: no cover
    pass

# --------------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
NOTEBOOK_DIR = ROOT / "notebooks"
OUTPUT_DIR = ROOT / "outputs"
CACHE_PATH = ROOT / "outputs" / "llm_cache.json"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

INPUT_COLUMNS = [
    "case_id", "modality", "body_part", "study_description",
    "patient_age_band", "patient_sex", "template_content", "dictation",
]

# ------------------------------------------------------------------ providers

# Both providers expose an OpenAI-compatible /chat/completions API, so one
# client covers both. `fallbacks` is walked in order so a single retired model
# id cannot kill a run.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        # Verified against GET /models on Groq (2026-09). The free tier is
        # aggressively rate-limited, so keep the chain short and the
        # concurrency low.
        "fallbacks": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ],
    },
    "opencode": {
        "base_url": "https://opencode.ai/zen/v1",
        "key_env": "OPENCODE_API_KEY",
        # NOTE: Zen model ids carry NO "opencode/" prefix on the wire.
        #
        # Verified 2026-09-27 against this account:
        #   * Paid ids (gpt-5.5, claude-sonnet-5, gemini-3.8-flash, ...) return
        #     402 "Insufficient account balance" -- the key is valid but the
        #     account has no credits.
        #   * The `-free` ids return 403 "OpenCode's free tier can only be used
        #     from within OpenCode", so they are not callable from a script.
        #   * `space-bunny-free` is the exception and does work from a script.
        # Set LLM_MODEL to any paid id once the account has credits.
        "default_model": "space-bunny-free",
        "fallbacks": [
            "space-bunny-free",
            "muse-spark-1.3-contributor-free",
            "nemotron-3-ultra-free",
            "big-pickle",
        ],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        # Verified on the OpenRouter :free catalogue. nemotron-3-super is the
        # default because it is the only free model that honours JSON mode and
        # answers in well under a second; gemma-4-31b:free frequently 429s.
        "default_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "fallbacks": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-4-31b-it:free",
            "qwen/qwen3.8-27b:free",
        ],
    },
}

DEFAULT_GUARD_LEVEL = "numbers"
# 0 shots, not 3. Both free tiers ration *prompt tokens*, and at 1 shot 6 of 40
# dev rows were rate-limited into a template fallback, which is worse than not
# prompting at all. Measured: RES 0.4366 at 0 shots vs 0.6193 unedited.
DEFAULT_N_SHOTS = 0
DEFAULT_DEV_SIZE = 160
DEV_SEED = 42


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def model_name(provider: str | None = None) -> str:
    override = os.getenv("LLM_MODEL", "").strip()
    if override:
        return override
    return PROVIDERS[provider or provider_name()]["default_model"]


def have_key(provider: str | None = None) -> bool:
    name = provider or provider_name()
    env = PROVIDERS[name]["key_env"]
    return bool(os.getenv(env, "").strip())

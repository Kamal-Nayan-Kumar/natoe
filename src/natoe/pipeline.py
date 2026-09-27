"""
Radiology dictation -> structured report pipeline.

Design (see notes in the accompanying notebook):

  1. RETRIEVE  2-3 similar train cases (same body_part / modality family) so the
     model sees the house style used for that exact template shape.
  2. EXTRACT   the LLM returns a JSON *patch*: only the FINDINGS fields that
     change, plus the IMPRESSION text. It never emits the whole report.
  3. GUARD     deterministic post-checks strip unsupported numbers /
     laterality / severity and any hallucinated field label.
  4. RENDER    Python assembles the report from the template, so field labels,
     field order and untouched normal statements are preserved byte-for-byte.

The LLM does the language work; the code does the bookkeeping. That split is
what keeps RES low, because RES is a template-edit metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter

import pandas as pd

from .config import PROVIDERS
from .res_scorer import (split_sections, parse_fields, normalize, tokenize,
                         canonical, _LIST_MARKER)

__all__ = [
    "PROVIDERS", "Retriever", "SYSTEM_PROMPT", "LLM", "MockLLM", "Cache",
    "clip", "build_user_prompt", "parse_json", "_sentences", "unsupported",
    "guard", "render", "gold_patch", "gold_json", "build_shots", "run_case",
    "run_dataset",
]

# --------------------------------------------------------------------------
# 1. Few-shot retrieval  (dependency-free TF-IDF)
# --------------------------------------------------------------------------

MAX_CHARS_TEMPLATE = 4000      # truncate the huge Lung-RADS boilerplate
MAX_CHARS_DICTATION = 6000

def _norm_key(s: str) -> str:
    return " ".join(tokenize(s))


class Retriever:
    """TF-IDF cosine over normalised tokens, cosine similarity.

    Dependency-free on purpose: the corpus is only a few hundred rows, so
    numpy-free sparse dot products are more than fast enough and keep the
    notebook self-contained.
    """

    def __init__(self, corpus: list[str]):
        self.docs = [_norm_key(c) for c in corpus]
        self.n = max(1, len(self.docs))
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d.split()))
        # smoothed inverse document frequency
        self.idf = {t: math.log(self.n / (1 + c)) + 1.0 for t, c in df.items()}
        self.max_idf = math.log(self.n) + 1.0
        self.vecs = [self._vec(d) for d in self.docs]

    def _vec(self, doc: str) -> dict[str, float]:
        v = {t: (1.0 + math.log(c)) * self.idf.get(t, self.max_idf)
             for t, c in Counter(doc.split()).items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def query(self, text: str, top_k: int = 3, exclude: set | None = None) -> list[int]:
        q = self._vec(_norm_key(text))
        scored = []
        for i, v in enumerate(self.vecs):
            if exclude and i in exclude:
                continue
            if len(q) > len(v):                      # iterate the smaller side
                s = sum(q[t] * v.get(t, 0.0) for t in q)
            else:
                s = sum(v[t] * q.get(t, 0.0) for t in v)
            scored.append((s, i))
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [i for _, i in scored[:top_k]]


# --------------------------------------------------------------------------
# 2. Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a radiology report editor. You convert a radiologist's dictation into an \
edit of a NORMAL reference template. The metric scores how faithfully you edit \
the template, so copying exact wording matters far more than being elegant.

Return ONE JSON object and nothing else:

{
  "fields": {
    "<EXACT TEMPLATE LABEL>": "<new content for that field>"
  },
  "impression": "<the full IMPRESSION section text>"
}

HARD RULES
1. Copy the radiologist's sentences VERBATIM into the matching field. Never \
paraphrase, never tidy grammar, never "improve" wording. Numbers, laterality \
and severity must match the dictation exactly.
2. Two normalisations, and only these two. Everything else is copied as-is:
   a) EXPAND spinal-level shorthand to the canonical hyphenated form: "c56" \
and "c6 7" and "c5-6, c7" all become "C5-C6", "C6-C7", "C5-C6 and C6-C7". \
Same for L3-4 -> L3-L4, T12-13 -> T12-T13. The reference reports do this in \
87% of cases.
   b) KEEP ordinary misspellings exactly as dictated. The reference reports \
reproduce the dictation's spelling errors verbatim in 100% of cases, so \
"ostearth", "andd", "dspaces" must appear in your output unchanged. Fixing a \
spelling error is scored as an edit. Do not expand abbreviations, do not \
reformat numbers, do not tidy grammar.
3. Only include a field in "fields" if you are CHANGING it. Fields the \
dictation says nothing about must be left out entirely — the renderer keeps \
the template's original text for them.
4. Use ONLY the field labels that appear in the template, spelled exactly as \
they appear (including slashes and spaces). Never invent a new field.
5. Route each finding to the field that owns that anatomy, and route it to \
exactly ONE field. An opacity goes in a lung field, an effusion in a pleural \
field, degenerative change in a bone or disc field, a fracture in a bone \
field. If the template has an "OTHER FINDINGS" field, use it only for \
findings that belong nowhere else. Getting this wrong is the single most \
expensive mistake you can make.
6. When a field is abnormal, DELETE the template's contradicting normal \
sentence (e.g. do not keep "The lungs are clear" next to an opacity) but KEEP \
any normal sentence that stays true and was not contradicted. Copy those kept \
normals out of the template word for word.
7. If the dictation explicitly negates something the template states as normal, \
write the negated form and keep the surrounding normal wording.
8. IMPRESSION rules, in priority order:
   a) If the dictation describes NO abnormality (it is normal, or it only \
contains technique/positioning notes such as "no acute traumatic process", \
"external marker projects over", "exposure adequate"), leave the template's \
IMPRESSION completely unchanged. Never put a technical note in the \
IMPRESSION.
   b) Otherwise the IMPRESSION is a short numbered or plain list of the \
important abnormal findings, built from the dictation's own summary wording.
9. Never add a finding, measurement, diagnosis or history that is not in the \
dictation. Do not comment on the images yourself.

EXAMPLE OF THE REQUIRED EDIT STYLE
Dictation: "no effusuon, infiltrates / mild thoracic spondylosis"
Template BONES field: "No acute osseous abnormality identified on this examination."
Edited BONES field: "Mild thoracic spondylosis is present. No acute osseous \
abnormality identified on this examination."
The new abnormality is prepended; the still-true normal sentence is kept.
Note the misspelling "effusuon" is left alone, per rule 2b.
"""


def build_user_prompt(case: dict, shots: list[dict]) -> str:
    parts = []
    for s in shots:
        parts.append(
            "### EXAMPLE\n"
            f"TEMPLATE:\n{clip(s['template_content'])}\n\n"
            f"DICTATION:\n{clip(s['dictation'])}\n\n"
            f"JSON OUTPUT:\n{s['_gold_json']}\n"
        )

    tpl = clip(case["template_content"])
    _f, display, _u = parse_fields(tpl)
    labels = [display[k] for k in _f]
    parts.append(
        "### NOW EDIT THIS CASE\n"
        f"modality: {case['modality']}\n"
        f"body_part: {case['body_part']}\n"
        f"study: {case['study_description']}\n"
        f"age_band: {case['patient_age_band']}\n"
        f"sex: {case['patient_sex']}\n\n"
        f"TEMPLATE FIELD LABELS (use these exactly): {json.dumps(labels)}\n\n"
        f"TEMPLATE:\n{tpl}\n\n"
        f"DICTATION:\n{clip(case['dictation'])}\n\n"
        "JSON OUTPUT:"
    )
    return "\n\n".join(parts)


def clip(text: str, n: int = MAX_CHARS_TEMPLATE) -> str:
    text = "" if text is None else str(text)
    if len(text) <= n:
        return text
    # keep the head (FINDINGS + IMPRESSION) and drop long trailing boilerplate
    return text[:n] + "\n...[truncated]"


# --------------------------------------------------------------------------
# 3. LLM client (OpenAI-compatible: Groq / OpenRouter / anything)
# --------------------------------------------------------------------------

# Reasoning models burn most of their output budget on chain-of-thought. For
# this task the reasoning is not the bottleneck, so keep it minimal.
REASONING_MODELS = ("gpt-oss", "nemotron", "qwen3")


def _is_fatal(exc: Exception) -> bool:
    """True when retrying the same request cannot help (bad model id, bad param)."""
    status = getattr(exc, "status_code", None)
    if status in (400, 401, 403, 404):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("model_not_found", "does not exist",
                                  "invalid_api_key", "not supported",
                                  "unsupported_parameter",
                                  "reasoning_effort is not supported"))


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or \
        "429" in str(exc) or "rate limit" in str(exc).lower()


def _backoff(exc: Exception, attempt: int) -> float:
    """Wait before retrying.

    Rate limits get a much longer wait, because the Groq token budget
    (8000 prompt tokens/min) needs a whole window to refill, and retrying
    early just burns the remaining quota.
    """
    if not _is_rate_limited(exc):
        return 2.0 * (2 ** attempt)
    delay = 20.0 * (2 ** attempt)
    resp = getattr(exc, "response", None)
    if resp is not None:
        hint = None
        try:
            hint = resp.headers.get("retry-after")
        except Exception:                                 # noqa: BLE001
            hint = None
        if hint:
            try:
                delay = max(delay, float(hint))
            except (TypeError, ValueError):
                pass
    return min(delay, 120.0)


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 reasoning_effort: str = "low"):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        cfg = PROVIDERS.get(self.provider)
        if cfg is None:
            raise ValueError(f"unknown provider {self.provider!r}; "
                             f"choose from {list(PROVIDERS)}")
        self.cfg = cfg
        key = os.getenv(cfg["key_env"], "").strip()
        if not key:
            raise RuntimeError(f"{cfg['key_env']} is not set. Put it in .env")
        self.api_key = key
        env_model = os.getenv("LLM_MODEL", "").strip()
        self.model = model or env_model or cfg["default_model"]
        self.chain = [self.model] + [m for m in cfg["fallbacks"]
                                     if m != self.model]
        self.reasoning_effort = (os.getenv("REASONING_EFFORT", reasoning_effort)
                                 or "").strip()
        # A hard socket timeout matters more than a generous one: a hung
        # request otherwise stalls a whole worker for minutes and the row
        # ends up on the template fallback.
        self.timeout = float(os.getenv("LLM_TIMEOUT", "90"))
        from openai import OpenAI
        self.client = OpenAI(base_url=cfg["base_url"], api_key=self.api_key,
                             timeout=self.timeout, max_retries=0)

    def _is_reasoning(self, model: str) -> bool:
        return any(k in model.lower() for k in REASONING_MODELS)

    def complete(self, prompt: str, system: str = SYSTEM_PROMPT,
                 max_tokens: int = 1400, temperature: float = 0.0,
                 attempts: int = 3) -> str:
        """Return raw model text.

        Retries the same model with exponential backoff (free tiers rate-limit
        aggressively under concurrency), then walks the fallback model list.
        Degrades gracefully along the way: drops JSON mode if the endpoint
        rejects it, and if a reasoning model spends its whole budget on
        chain-of-thought and returns no `content`, salvages the JSON object out
        of the `reasoning` field instead.
        """
        last: Exception | None = None
        errors: list[str] = []
        for model in self.chain:
            for json_mode in (True, False):
                for attempt in range(attempts):
                    try:
                        kw: dict = dict(
                            model=model,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if json_mode:
                            kw["response_format"] = {"type": "json_object"}
                        if self._is_reasoning(model) and self.reasoning_effort:
                            kw["reasoning_effort"] = self.reasoning_effort
                        msg = self.client.chat.completions.create(
                            **kw).choices[0].message
                        content = (getattr(msg, "content", None) or "").strip()
                        if content:
                            return content
                        reasoning = (getattr(msg, "reasoning", None) or "").strip()
                        if reasoning:
                            return _last_json_object(reasoning) or reasoning
                        last = RuntimeError("empty content and reasoning")
                        break              # nothing to retry
                    except Exception as e:       # noqa: BLE001
                        last = e
                        if _is_fatal(e):
                            break              # this model/flag combo is wrong
                        time.sleep(_backoff(e, attempt))
                errors.append(f"{model}[json={json_mode}]: "
                              f"{type(last).__name__}: {str(last)[:120]}")
        raise RuntimeError("all models failed -> " + " | ".join(errors[:4]))


def _last_json_object(text: str) -> str | None:
    """Pull the last balanced {...} out of a blob, ignoring braces in strings."""
    depth = start = 0
    best = None
    in_str = escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                best = text[start:i + 1]
    return best


class MockLLM:
    """Offline stand-in so the notebook can be smoke-tested without a key.

    Crude but real: it routes each dictation sentence to the template field
    whose anatomy keywords it matches best. The score is meaningless -- it only
    proves the plumbing (prompt -> JSON -> guard -> render) runs end to end.
    """

    provider, cfg, api_key = "mock", {}, ""
    model, chain = "mock", ["mock"]

    @staticmethod
    def _keys(label: str) -> set[str]:
        stop = {"no", "the", "and", "of", "is", "are", "normal", "unremarkable",
                "finding", "findings", "structures", "space", "spaces"}
        return {w for w in re.findall(r"[a-z]{4,}", label.lower())
                if w not in stop}

    def complete(self, prompt, system=None, max_tokens=2000, temperature=0.0):
        tpl = re.search(r"TEMPLATE:\n(.*?)\n\nDICTATION:", prompt, re.S)
        dct = re.search(r"DICTATION:\n(.*?)\n\nJSON OUTPUT:", prompt, re.S)
        labs = re.search(r"FIELD LABELS \(use these exactly\): (\[.*?\])",
                         prompt, re.S)
        if not (dct and labs):
            return '{"fields": {}, "impression": ""}'
        sents = _sentences(dct.group(1))
        fields = {}
        for lab in json.loads(labs.group(1)):
            keys = self._keys(lab)
            if not keys:
                continue
            best, hits_best = "", 0
            for s in sents:
                hits = len(keys & set(re.findall(r"[a-z]{4,}", s.lower())))
                if hits > hits_best:
                    best, hits_best = s, hits
            if hits_best >= 1:
                fields[lab] = best
        return json.dumps({"fields": fields,
                           "impression": "\n".join(sents[:3])},
                          ensure_ascii=False)


# --------------------------------------------------------------------------
# 4. Parse + guard the model patch
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)

LATERALITY = {"right", "left", "bilateral", "unilateral", "ipsilateral",
              "contralateral"}
# Comparative qualifiers only. Deliberately excludes "acute", "chronic" and
# "significant": they appear in almost every normal template sentence, so
# flagging them would delete faithful text.
SEVERITY = {"mild", "moderate", "severe", "minimal", "minimally", "small",
            "large", "tiny", "trace", "marked", "extensive", "subtle",
            "slight", "slightly", "grossly", "borderline"}


def parse_json(txt: str) -> dict:
    txt = _FENCE.sub("", txt or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def _sentences(text: str) -> list[str]:
    """Split into sentences line by line, keeping list items ("1. foo") glued
    to their first sentence so numbered IMPRESSION lines survive a round trip.
    """
    out: list[str] = []
    for line in (text or "").split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        head = ""
        m = _LIST_MARKER.match(line)
        if m:
            head, line = m.group(0), line[m.end():]
        parts = [p.strip() for p in
                 re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])", line) if p.strip()]
        for i, p in enumerate(parts):
            out.append(head + p if i == 0 else p)
    return out


def unsupported(sent: str, allowed: set[str], level: str = "numbers") -> list[str]:
    """Critical tokens in `sent` that have no support in `allowed`.

    `allowed` is the dictation's token set UNION the field's original template
    text, so faithfully preserved normal sentences survive.

    level:
      "off"     - no check
      "numbers" - only new measurements are flagged. A measurement is a
                  concrete factual claim, so inventing one is the clearest
                  form of unsupported content. This is the default.
      "strict"  - also flag new laterality / severity words.
    Negation is never used as a trigger: the template legitimately says
    "No pneumothorax." even when the dictation never mentioned pneumothorax.
    """
    if level == "off":
        return []
    out = []
    for t in tokenize(sent):
        if t in allowed:
            continue
        if any(c.isdigit() for c in t):
            out.append(t)                                   # measurements
        elif level == "strict" and (t in LATERALITY or t in SEVERITY):
            out.append(t)
    return out


def guard(patch: dict, case: dict, level: str = "numbers") -> dict:
    """Drop hallucinated field labels and unsupported sentences."""
    tpl_fields, _tpl_display, _ = parse_fields(
        split_sections(case["template_content"])[0])
    dict_tok = set(tokenize(case["dictation"]))
    tpl_imp_tok = set(tokenize(split_sections(case["template_content"])[1]))
    drops = []

    clean = {}
    for label, content in (patch.get("fields") or {}).items():
        key = canonical(label)
        if key not in tpl_fields:               # tolerate slug/case slips
            cands = [k for k in tpl_fields if key and (key in k or k in key)]
            if cands:
                key = min(cands, key=len)
            else:
                drops.append(f"label:{label}")
                continue
        allowed = dict_tok | set(tokenize(tpl_fields[key]))
        sents = _sentences(content)
        keep = [s for s in sents if not unsupported(s, allowed, level)]
        for s in sents:
            if s not in keep:
                drops.append(f"unsupported:{s[:40]}")
        if keep:
            clean[key] = " ".join(keep)

    imp = patch.get("impression") or ""
    imp_allowed = dict_tok | tpl_imp_tok
    imp_sents = _sentences(imp)
    imp_keep = [s for s in imp_sents if not unsupported(s, imp_allowed, level)]
    imp_clean = "\n".join(imp_keep)
    if imp.strip() and not imp_keep:
        drops.append("impression_fully_unsupported")

    return {"fields": clean, "impression": imp_clean, "drops": drops}


# --------------------------------------------------------------------------
# 5. Render the final report (pure Python, template-faithful)
# --------------------------------------------------------------------------

def render(case: dict, clean: dict) -> str:
    tpl_fields, tpl_display, _ = parse_fields(
        split_sections(case["template_content"])[0])
    tpl_imp = split_sections(case["template_content"])[1]

    lines = []
    for key in tpl_fields:                      # template order, always
        label = tpl_display[key]
        content = clean["fields"].get(key, tpl_fields[key]).strip()
        lines.append(f"{label}: {content}".rstrip())

    impression = (clean["impression"] or "").strip() or tpl_imp.strip()
    return "FINDINGS:\n" + "\n".join(lines) + "\n\nIMPRESSION:\n" + impression


# --------------------------------------------------------------------------
# 6. Gold patch builder (used to create few-shot exemplars from train.csv)
# --------------------------------------------------------------------------

def gold_patch(row) -> dict:
    ref_fields, _d, _u = parse_fields(split_sections(row.report)[0])
    tpl_fields, _td, _tu = parse_fields(split_sections(row.template_content)[0])
    changed = {}
    for k, v in ref_fields.items():
        if k in tpl_fields and normalize(v).strip() != \
                normalize(tpl_fields[k]).strip():
            changed[k] = v
    return {
        "fields": changed,
        "impression": split_sections(row.report)[1].strip(),
    }


def gold_json(row) -> str:
    p = gold_patch(row)
    # keep exemplars short: drop fields whose gold edit is huge
    p["fields"] = {k: v for k, v in list(p["fields"].items())[:6]}
    return json.dumps(p, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------
# 7. Runner (with an on-disk cache so prompt iteration is cheap)
# --------------------------------------------------------------------------

class Cache:
    """On-disk response cache, safe for concurrent writers.

    Writes are serialised by a lock and land via an atomic replace, so a run
    that is killed mid-write can never leave a truncated cache behind.
    """

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, str] = {}
        if path and os.path.exists(path):
            try:
                with open(path) as fh:
                    self.data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self.data = {}

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()

    def get(self, k):
        with self._lock:
            return self.data.get(k)

    def put(self, k, v):
        # Never cache a failure. A transient 429 that got cached would poison
        # every later run of the same config.
        if not v:
            return
        with self._lock:
            self.data[k] = v
            if not self.path:
                return
            tmp = f"{self.path}.tmp"
            try:
                with open(tmp, "w") as fh:
                    json.dump(self.data, fh)
                os.replace(tmp, self.path)          # atomic
            except OSError:
                pass


def build_shots(retriever: Retriever, train: pd.DataFrame, case: dict,
                n_shots: int = 3) -> list[dict]:
    """Nearest training cases, as ready-to-print exemplars."""
    if n_shots <= 0:
        return []
    query = f"{case['body_part']} {case['modality']} {case['study_description']} " \
            f"{case['template_content']} {case['dictation']}"
    idx = retriever.query(query, top_k=n_shots)
    shots = []
    for i in idx:
        row = train.iloc[i]
        shots.append({
            "template_content": row.template_content,
            "dictation": row.dictation,
            "_gold_json": gold_json(row),
        })
    return shots


def run_case(llm, case: dict, shots: list[dict],
             guard_level: str = "numbers") -> tuple[str, dict]:
    prompt = build_user_prompt(case, shots)
    raw = llm.complete(prompt)
    try:
        patch = parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        patch = {}                      # unparseable -> fall back to template
    clean = guard(patch, case, guard_level)
    return render(case, clean), clean


def run_dataset(llm, df: pd.DataFrame, retriever: Retriever,
                train: pd.DataFrame, n_shots: int = 3,
                guard_level: str = "numbers", cache_path: str | None = None,
                progress: bool = True, workers: int = 2,
                retries: int = 2) -> list[str]:
    """Predict every row. Responses are cached on disk keyed by
    (model, reasoning_effort, guard_level, n_shots, system_prompt, prompt) so
    prompt iteration is nearly free.

    `workers` and `retries` exist because of rate limiting, not throughput.
    Free tiers ration *prompt tokens* (Groq: 8000/min), so a burst of parallel
    few-shot requests gets 429s and silently degrades the score to
    "template unedited". Failing loudly and retrying slowly is worth far more
    than raw speed here.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache = Cache(cache_path)
    cases, prompts = [], []
    for _, r in df.iterrows():
        case = {"template_content": r.template_content, "dictation": r.dictation,
                "modality": r.modality, "body_part": r.body_part,
                "study_description": r.study_description,
                "patient_age_band": r.patient_age_band,
                "patient_sex": r.patient_sex}
        shots = build_shots(retriever, train, case, n_shots)
        prompt = build_user_prompt(case, shots)
        cases.append(case)
        prompts.append(prompt)

    failures: list[str] = []

    def fetch(args):
        i, prompt = args
        # The system prompt MUST be part of the key: otherwise editing the
        # prompt rules would silently reuse every stale cached response.
        ck = Cache.key(llm.model, llm.reasoning_effort, guard_level, n_shots,
                       SYSTEM_PROMPT, prompt)
        raw = cache.get(ck)
        if raw is None:
            try:
                raw = llm.complete(prompt)
            except Exception as e:                       # noqa: BLE001
                failures.append(f"row {i}: {type(e).__name__}: {str(e)[:160]}")
                return ""
            cache.put(ck, raw)
        return raw

    t0 = time.time()
    n = len(cases)
    indices = list(range(1, n + 1))

    def sweep(idxs: list[int], label: str) -> None:
        if not idxs:
            return
        if progress:
            print(f"  {label}: {len(idxs)} row(s)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, raw in enumerate(pool.map(fetch, [(i, prompts[i - 1])
                                                       for i in idxs]), 1):
                raws[i - 1] = raw
                if progress and (done % 10 == 0 or done == len(idxs)):
                    _tick(done, len(idxs), t0)

    sweep(indices, "pass 1")
    for attempt in range(1, retries + 1):
        missing = [i for i in indices if not raws[i - 1]]
        if not missing:
            break
        # Failures are never cached, so a retry costs a fresh call and a
        # second chance is always worth taking.
        time.sleep(30.0 * attempt)
        failures.clear()
        sweep(missing, f"retry {attempt}")

    if progress:
        print()
    if failures:
        print(f"  WARNING: {len(failures)} row(s) exhausted retries and fell "
              f"back to the template. First 5:")
        for f in failures[:5]:
            print(f"    {f}")

    outs = []
    for case, raw in zip(cases, raws):
        try:
            patch = parse_json(raw) if raw else {}
        except (ValueError, json.JSONDecodeError):
            patch = {}                   # unparseable -> fall back to template
        outs.append(render(case, guard(patch, case, guard_level)))
    return outs


def _tick(done: int, total: int, t0: float) -> None:
    rate = done / max(1e-9, time.time() - t0)
    print(f"  {done}/{total}  {rate:.2f} rows/s  "
          f"eta {(total - done) / max(1e-9, rate):.0f}s")

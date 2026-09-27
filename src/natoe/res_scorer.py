"""
Local re-implementation of the RES (Radiology Edit Score) leaderboard metric.

Written from the competition description in detail.md. The exact leaderboard
scorer is private, so a few details are best-effort guesses (flagged with
`# ASSUMPTION`). Use it for *relative* comparison of pipeline variants, and
keep absolute numbers in mind as optimistic/pessimistic.

RES_case = 0.65 * F + 0.35 * I
  F = field-aware FINDINGS score
  I = weighted word edit distance over the whole IMPRESSION section
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# 1. Tokenisation / normalisation  (section: "Text normalization")
# --------------------------------------------------------------------------

# ASSUMPTION: the exact function-word and critical-word lists are not published.
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "these", "this", "those", "to", "was", "were", "which",
    "with", "within", "there", "here", "notably", "again", "also",
}

CRITICAL_WORDS = {
    # negation
    "no", "not", "without", "absent", "negative", "unremarkable", "clear",
    # laterality
    "right", "left", "bilateral", "unilateral", "ipsilateral", "contralateral",
    # severity
    "mild", "moderate", "severe", "minimal", "minimally", "small", "large",
    "tiny", "trace", "marked", "extensive", "subtle", "significant", "slight",
    "slightly", "grossly", "borderline", "chronic", "acute",
    # laterality-ish / descriptors the spec calls clinically critical
    "focal", "diffuse", "stable", "new", "interval",
    # units
    "mm", "cm", "mm2", "cm2",
}

# ASSUMPTION: list of "common unit spellings" the description says are
# standardised. Extended a little for robustness.
UNIT_CANON = {
    "millimeters": "mm", "millimetres": "mm", "millimeter": "mm",
    "millimetre": "mm",
    "centimeters": "cm", "centimetres": "cm", "centimeter": "cm",
    "centimetre": "cm",
}

_LIST_MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*+•‣▪])\s+")
_SIGNED_NUM = re.compile(r"^[+-]\d")
_PUNCT = re.compile(r"[^\w\s+/-]|(?<=\w)/|/(?=\w)")


def normalize(text: str) -> str:
    """Lowercase, unicode-normalise, standardise units, strip list markers."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text).lower()
    for long, short in UNIT_CANON.items():
        text = re.sub(rf"\b{long}\b", short, text)
    out_lines = []
    for line in text.split("\n"):
        prev = None
        while prev != line:                      # markers can stack ("1. - ")
            prev = line
            line = _LIST_MARKER.sub("", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def tokenize(text: str) -> list[str]:
    """Normalised tokens: punctuation stripped, list markers removed, hyphens
    between letters deleted ("air-space" -> "airspace") but hyphens touching a
    digit act as separators ("L5-S1" -> "l 5 s 1") so that glued and spaced
    spinal-level spellings compare equal. Letter/number boundaries are split.
    """
    text = normalize(text)
    # Keep + and - (the punctuation class above preserves them), stash signed
    # measurements next, then apply hyphen rules. The placeholder must be
    # letters-only so the letter/number boundary splits ignore it.
    text = _PUNCT.sub(" ", text)
    signed: list[str] = []

    def _stash(m):
        signed.append(m.group(0).replace(" ", ""))
        return " zzsig" + "a" * len(signed) + " "

    text = re.sub(r"[+\-]\s*\d+(?:\.\d+)?", _stash, text)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", "", text)   # air-space -> airspace
    text = re.sub(r"[+\-]", " ", text)                  # L5-S1 -> L 5 S 1
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)       # t9 -> t 9
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)       # 7th -> 7 th
    out = []
    for t in text.split():
        m = re.fullmatch(r"zzsig(a*)", t)
        out.append(signed[len(m.group(1)) - 1] if m else t)  # "+3" as one token
    return [t for t in out if t]


# --------------------------------------------------------------------------
# 2. Weighted word edit distance
# --------------------------------------------------------------------------

W_CRITICAL, W_CONTENT, W_FUNCTION = 4.0, 2.0, 0.25


def token_weight(tok: str) -> float:
    if tok in CRITICAL_WORDS or tok in UNIT_CANON.values():
        return W_CRITICAL
    if tok in FUNCTION_WORDS:
        return W_FUNCTION
    if any(ch.isdigit() for ch in tok):
        return W_CRITICAL                       # measurements
    return W_CONTENT


def _total_weight(tokens: list[str]) -> float:
    return sum(token_weight(t) for t in tokens)


def weighted_word_edit(ref: str, sub: str) -> float:
    """Ordered weighted Levenshtein, normalised by the larger token weight."""
    a, b = tokenize(ref), tokenize(sub)
    wa, wb = _total_weight(a), _total_weight(b)
    denom = max(wa, wb)
    if denom == 0:
        return 0.0
    n, m = len(a), len(b)
    # cost[i][j] = min cost to transform a[:i] into b[:j]
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + token_weight(b[j - 1])          # insert
    for i in range(1, n + 1):
        cur = [prev[0] + token_weight(a[i - 1])]                 # delete
        wa_i = token_weight(a[i - 1])
        for j in range(1, m + 1):
            sub_cost = 0.0 if a[i - 1] == b[j - 1] else max(
                wa_i, token_weight(b[j - 1])
            )
            cur.append(min(prev[j] + wa_i,      # delete
                           cur[j - 1] + token_weight(b[j - 1]),   # insert
                           prev[j - 1] + sub_cost))               # substitute
        prev = cur
    return min(1.0, prev[m] / denom)


# --------------------------------------------------------------------------
# 3. Section / field parsing
# --------------------------------------------------------------------------

# Labels are matched case-insensitively: real templates contain BOTH "BONES:"
# and "Bones:", and the scorer lowercases everything before comparing, so the
# two must align. Getting this wrong silently drops whole fields.
_LABEL = re.compile(r"^([A-Za-z][A-Za-z0-9 /\-&.,()']{1,70}):\s*(.*)$")
RESERVED_LABELS = {"FINDINGS", "IMPRESSION"}
_LABEL_STRIP = re.compile(r"[^A-Z0-9]+")


def canonical(label: str) -> str:
    """Case/space/punctuation-insensitive key for aligning fields by label."""
    return _LABEL_STRIP.sub("", (label or "").upper())


def split_sections(text: str) -> tuple[str, str, str]:
    """-> (findings_block, impression_block, trailing_block)"""
    text = text if isinstance(text, str) else ""
    m = re.search(r"(?m)^\s*FINDINGS\s*:", text)
    if not m:
        return "", text, ""
    head = text[: m.start()]
    rest = text[m.end():]
    mi = re.search(r"(?m)^\s*IMPRESSION\s*:", rest)
    if not mi:
        return rest, "", ""
    findings = rest[: mi.start()]
    impression = rest[mi.end():]
    # Anything after IMPRESSION that is not part of it (e.g. LUNG-RADS block)
    # is treated as trailing and is ignored by the scorer.
    return findings, impression, ""


def parse_fields(block: str) -> tuple[dict[str, str], dict[str, str], str]:
    """Split a FINDINGS block.

    Returns (fields, display, unlabelled) where
      fields    : {canonical_label: content}
      display   : {canonical_label: label exactly as written}
      unlabelled: free text sitting directly under FINDINGS:
    """
    fields: dict[str, str] = {}
    display: dict[str, str] = {}
    unlabelled: list[str] = []
    current: str | None = None
    buf: list[str] = []

    def flush():
        if current is not None:
            fields[current] = " ".join(x.strip() for x in buf).strip()

    for raw in (block or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _LABEL.match(line)
        if m and m.group(1).strip().upper() not in RESERVED_LABELS:
            flush()
            current = canonical(m.group(1))
            display.setdefault(current, m.group(1).strip())
            buf = [m.group(2)]
        elif current is None:
            unlabelled.append(line)
        else:
            buf.append(line)
    flush()
    return fields, display, " ".join(unlabelled).strip()


# --------------------------------------------------------------------------
# 4. RES
# --------------------------------------------------------------------------

def findings_score(ref_f: dict[str, str], sub_f: dict[str, str],
                   tpl_f: dict[str, str], ref_disp: dict | None = None
                   ) -> tuple[float, dict]:
    """
    F = sum(field_weight * field_word_edit) / sum(field_weight)

    field_weight = 3 if the reference field differs from the template field
                    1 if the reference field is unchanged from the template
    Missing expected field -> compare reference content against "".
    Unexpected fields / unlabelled content -> extra penalty (ASSUMPTION: added
    as a weight-1 pseudo-field, which mirrors "additional penalties").
    """
    ref_disp = ref_disp or {}
    num = den = 0.0
    detail = {}
    for label, ref_txt in ref_f.items():
        tpl_txt = tpl_f.get(label, "")
        changed = normalize(ref_txt).strip() != normalize(tpl_txt).strip()
        w = 3.0 if changed else 1.0
        sub_txt = sub_f.get(label)
        e = 1.0 if sub_txt is None else weighted_word_edit(ref_txt, sub_txt)
        num += w * e
        den += w
        detail[label] = {"w": w, "edit": round(e, 3),
                         "missing": sub_txt is None,
                         "ref": ref_txt[:90],
                         "ref_display": ref_disp.get(label, label)}

    # unexpected fields: penalised as extra content
    for label, sub_txt in sub_f.items():
        if label in ref_f:
            continue
        e = weighted_word_edit(sub_txt, "") if sub_txt.strip() else 0.0
        num += 1.0 * e
        den += 1.0
        detail[label] = {"w": 1.0, "edit": round(e, 3), "unexpected": True,
                         "sub": sub_txt[:90]}

    if den == 0:
        return 0.0, detail
    return min(1.0, num / den), detail


def res_case(ref_report: str, sub_report: str, template: str) -> dict:
    r_fb, r_im, _ = split_sections(ref_report)
    s_fb, s_im, _ = split_sections(sub_report)
    t_fb, _t_im, _t_tr = split_sections(template)
    ref_f, ref_disp, _ = parse_fields(r_fb)
    sub_f, _sd, sub_unlab = parse_fields(s_fb)
    tpl_f, _td, _ = parse_fields(t_fb)

    F, detail = findings_score(ref_f, sub_f, tpl_f, ref_disp)
    I = weighted_word_edit(r_im, s_im)
    return {"F": F, "I": I, "RES": 0.65 * F + 0.35 * I, "fields": detail,
            "unlabelled": sub_unlab}


def res_dataset(refs, subs, templates) -> dict:
    """refs/subs/templates: pandas Series aligned by index."""
    rows = [res_case(r, s, t) for r, s, t in zip(refs, subs, templates)]
    F = sum(x["F"] for x in rows) / len(rows)
    I = sum(x["I"] for x in rows) / len(rows)
    return {"RES": 0.65 * F + 0.35 * I, "F": F, "I": I, "rows": rows}


if __name__ == "__main__":
    ref = """FINDINGS:
LUNGS: Mild right basilar airspace opacity. No pulmonary edema.
PLEURA: Small right pleural effusion. No pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
Mild right basilar airspace opacity and small right pleural effusion."""
    tpl = """FINDINGS:
LUNGS: No focal airspace opacity or pulmonary edema.
PLEURA: No pleural effusion or pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
No acute cardiopulmonary abnormality."""
    print("perfect ->", res_case(ref, ref, tpl)["RES"])
    print("template as-is ->", round(res_case(ref, tpl, tpl)["RES"], 4))
    print("dictation only ->", round(res_case(ref, "mild right basilar "
          "opacity, small right pleural effusion", tpl)["RES"], 4))

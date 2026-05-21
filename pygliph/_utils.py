"""Internal helpers shared across pygliph modules."""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

_AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")
_AA_VALID_O = set("ACDEFGHIKLMNOPQRSTUVWY")


def format_e1(x: float) -> str:
    """Reproduce R's ``formatC(x, digits = 1, format = "e")``.

    Produces strings such as ``"1.2e-03"`` / ``"0.0e+00"``.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    s = f"{float(x):.1e}"
    # Python: 1.2e-03 ; R uses at least 2 exponent digits too -> match.
    mant, exp = s.split("e")
    sign = exp[0]
    digits = exp[1:].lstrip("0") or "0"
    if len(digits) < 2:
        digits = digits.zfill(2)
    return f"{mant}e{sign}{digits}"


def as_numeric_format_e1(x: float) -> float:
    """``as.numeric(formatC(x, digits = 1, format = "e"))``."""
    return float(format_e1(x))


def is_aa_sequence(s: str, allow_extended: bool = False) -> bool:
    """Whether ``s`` contains only one-letter amino-acid codes."""
    alphabet = _AA_VALID_O if allow_extended else _AA_VALID
    return len(s) > 0 and all(c in alphabet for c in s.upper())


def filter_aa_sequences(seqs: Iterable[str], allow_extended: bool = False):
    """Keep only valid amino-acid sequences (mask returned alongside)."""
    out, mask = [], []
    for s in seqs:
        ok = is_aa_sequence(str(s), allow_extended)
        mask.append(ok)
        if ok:
            out.append(str(s))
    return out, np.asarray(mask, dtype=bool)


_CF_RE = re.compile(r"^C.*F$")


def starts_C_ends_F(s: str) -> bool:
    return bool(_CF_RE.match(s))


def hamming(a: str, b: str) -> int:
    """Hamming distance; ``inf`` for unequal lengths (as stringdist does)."""
    if len(a) != len(b):
        return 10 ** 9
    return sum(1 for x, y in zip(a, b) if x != y)


def round_r(x: float, digits: int = 0) -> float:
    """Round half-to-even, matching R's ``round``."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    factor = 10 ** digits
    return float(np.round(x * factor) / factor)

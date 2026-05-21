"""Continuous and discontinuous CDR3 motif enumeration.

Port of turboGliph's ``find_motifs`` (R/find_motifs.R).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["find_motifs", "count_kmers"]


def count_kmers(seqs: Iterable[str], q: int) -> Counter:
    """Count every length-``q`` continuous substring occurrence.

    Mirrors ``stringdist::qgrams`` summed across sequences: every position
    of a k-mer in every sequence contributes one count (so a k-mer that
    occurs twice in one sequence is counted twice).
    """
    counts: Counter = Counter()
    for s in seqs:
        n = len(s)
        for i in range(n - q + 1):
            counts[s[i:i + q]] += 1
    return counts


def find_motifs(
    seqs: Sequence[str],
    q: Sequence[int] = (2, 3, 4),
    kmer_mindepth: int | None = None,
    discontinuous: bool = False,
) -> pd.DataFrame:
    """Find and count continuous (and optional discontinuous) motifs.

    Parameters
    ----------
    seqs
        Sequences whose motifs are identified and quantified.
    q
        Motif lengths to search for. Default ``(2, 3, 4)``.
    kmer_mindepth
        If given, motifs observed fewer than this many times are dropped.
    discontinuous
        If ``True``, also enumerate motifs of length ``q`` with one
        internal position replaced by a wildcard ``"."`` (built from
        continuous motifs of length ``q + 1``).

    Returns
    -------
    pandas.DataFrame
        Two columns: ``motif`` and ``V1`` (the frequency), matching the
        R function's output column names.
    """
    seqs = [str(s) for s in seqs]
    q = list(q)
    all_q = sorted(set(q) | {x + 1 for x in q}) if discontinuous else list(q)

    rows: list[tuple[str, int]] = []
    for i in all_q:
        cont = count_kmers(seqs, i)
        cont_items = list(cont.items())
        if kmer_mindepth is not None:
            cont_items = [(m, c) for m, c in cont_items if c >= kmer_mindepth]

        if i in q:
            rows.extend(cont_items)

        if discontinuous and i in {x + 1 for x in q}:
            # discontinuous motifs: replace position j (1-based 2..i-1)
            for j in range(2, i):
                disc: Counter = Counter()
                for m, c in cont.items():
                    dm = m[:j - 1] + "." + m[j:]
                    disc[dm] += c
                disc_items = list(disc.items())
                if kmer_mindepth is not None:
                    disc_items = [(m, c) for m, c in disc_items
                                  if c >= kmer_mindepth]
                rows.extend(disc_items)

    if not rows:
        return pd.DataFrame({"motif": pd.Series(dtype=str),
                             "V1": pd.Series(dtype=int)})
    df = pd.DataFrame(rows, columns=["motif", "V1"])
    return df

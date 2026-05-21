"""Built-in datasets and reference resources for pygliph.

These are exact exports of the data objects shipped with the R package
``turboGliph`` (``gliph_input_data``, ``reference_list[["gliph_reference"]]``,
``ref_cluster_sizes``, ``gTRB``, ``BlosumVec``).
"""
from __future__ import annotations

import gzip
import os
from functools import lru_cache

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _path(name: str) -> str:
    return os.path.join(_DATA_DIR, name)


@lru_cache(maxsize=None)
def load_gliph_input_data() -> pd.DataFrame:
    """Return ``gliph_input_data`` -- ~2000 TCRs of known specificity.

    Equivalent to ``utils::data("gliph_input_data")`` in turboGliph.
    """
    df = pd.read_csv(_path("gliph_input_data.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    return df


@lru_cache(maxsize=None)
def load_reference_db(name: str = "gliph_reference") -> pd.DataFrame:
    """Return a naive reference repertoire as a DataFrame ``[CDR3b, TRBV]``.

    Only ``"gliph_reference"`` (162,165 CDR3b sequences) ships with the
    package, mirroring turboGliph.
    """
    if name != "gliph_reference":
        raise ValueError(
            "Only 'gliph_reference' is bundled. Pass a DataFrame for custom "
            "reference databases."
        )
    with gzip.open(_path("gliph_reference_refseqs.tsv.gz"), "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False)
    return df


@lru_cache(maxsize=None)
def load_vgene_ref_frequencies() -> pd.DataFrame:
    """V-gene usage frequencies in the naive reference repertoire."""
    return pd.read_csv(_path("gliph_reference_vgene_freq.tsv"), sep="\t")


@lru_cache(maxsize=None)
def load_cdr3_length_ref_frequencies() -> pd.DataFrame:
    """CDR3b length frequencies in the naive reference repertoire."""
    return pd.read_csv(_path("gliph_reference_cdr3_length_freq.tsv"), sep="\t")


@lru_cache(maxsize=None)
def load_ref_cluster_sizes(kind: str = "original") -> pd.DataFrame:
    """Cluster-size probability table used by the network-size score.

    Parameters
    ----------
    kind
        ``"original"`` (constant across sample sizes, as in the original
        GLIPH) or ``"simulated"`` (sample-size dependent, estimated by the
        turboGliph authors).
    """
    if kind not in ("original", "simulated"):
        raise ValueError("kind must be 'original' or 'simulated'")
    return pd.read_csv(_path(f"ref_cluster_sizes_{kind}.tsv"), sep="\t")


@lru_cache(maxsize=None)
def load_gtrb() -> dict:
    """Germline TRB CDR3 fragments (``gTRBV``, ``gTRBD``, ``gTRBJ``).

    Used by GLIPH2 to detect non-germline (N/P) encoded residues.
    """
    out = {}
    for gene in ("gTRBV", "gTRBD", "gTRBJ"):
        df = pd.read_csv(_path(f"{gene}.tsv"), sep="\t")
        out[gene] = df
    return out


@lru_cache(maxsize=None)
def load_blosum_vec() -> frozenset:
    """Amino-acid pairs with a non-negative BLOSUM62 score.

    Returns a ``frozenset`` of two-letter strings, e.g. ``"AA"``, ``"CA"``.
    """
    with open(_path("blosum_vec.txt")) as fh:
        return frozenset(line.strip() for line in fh if line.strip())

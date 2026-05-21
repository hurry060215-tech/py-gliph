"""pygliph -- pure-Python port of the R package turboGliph.

Implements GLIPH (Glanville et al., Nature 2017) and GLIPH2 (Huang et al.,
Nat. Biotechnol. 2020): grouping of T-cell receptors into specificity
groups by shared CDR3 motifs.

Main entry points
-----------------
- :func:`gliph2`            -- the GLIPH2 algorithm (Fisher-test local motifs).
- :func:`turbo_gliph`       -- the original GLIPH algorithm (resampling).
- :func:`gliph_combined`    -- configurable hybrid of the two.
- :func:`cluster_scoring`   -- per-cluster enrichment scoring.
- :func:`find_motifs`       -- continuous / discontinuous motif enumeration.
- :func:`de_novo_TCRs`      -- de-novo CDR3 generation from a group.
- :func:`plot_network`      -- specificity-group network visualisation.
- :func:`load_gliph_output` / :func:`save_gliph_output` -- result I/O.

Datasets
--------
``datasets`` exposes the bundled ``gliph_input_data``, the GLIPH naive
reference repertoire and the supporting frequency / germline tables.
"""
from __future__ import annotations

from . import datasets
from .combined import gliph_combined
from .de_novo import de_novo_TCRs
from .gliph2 import gliph2
from .io import load_gliph_output, save_gliph_output
from .motifs import find_motifs
from .plotting import plot_network
from .scoring import cluster_scoring
from .turbo_gliph import turbo_gliph

__version__ = "0.1.0"

__all__ = [
    "gliph2",
    "turbo_gliph",
    "gliph_combined",
    "cluster_scoring",
    "find_motifs",
    "de_novo_TCRs",
    "plot_network",
    "load_gliph_output",
    "save_gliph_output",
    "datasets",
    "__version__",
]

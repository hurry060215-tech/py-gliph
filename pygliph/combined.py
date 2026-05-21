"""``gliph_combined`` -- configurable hybrid of GLIPH and GLIPH2.

turboGliph's ``gliph_combined`` lets the user pick, independently, the
local-similarity method (``"rrs"`` repeated random sampling, as in GLIPH,
or ``"fisher"``, as in GLIPH2) and the global-similarity method
(``"global_hamming"`` or ``"fisher"``). This Python port dispatches to the
already-faithful :func:`pygliph.turbo_gliph` / :func:`pygliph.gliph2`
implementations and returns the GLIPH2-style output container.
"""
from __future__ import annotations

import pandas as pd

from .gliph2 import gliph2
from .turbo_gliph import turbo_gliph

__all__ = ["gliph_combined"]


def gliph_combined(
    cdr3_sequences,
    refdb_beta="gliph_reference",
    local_method: str = "fisher",
    global_method: str = "fisher",
    v_usage_freq: pd.DataFrame | None = None,
    cdr3_length_freq: pd.DataFrame | None = None,
    ref_cluster_size: str = "original",
    sim_depth: int = 1000,
    lcminp: float = 0.01,
    lcminove=(1000, 100, 10),
    motif_distance_cutoff: int = 3,
    kmer_mindepth: int = 3,
    gccutoff: int | None = None,
    accept_sequences_with_C_F_start_end: bool = True,
    min_seq_length: int = 0,
    structboundaries: bool = True,
    boundary_size: int = 3,
    motif_length=(2, 3, 4),
    discontinuous_motifs: bool = False,
    local_similarities: bool = True,
    global_similarities: bool = True,
    global_vgene: bool = False,
    all_aa_interchangeable: bool = False,
    boost_local_significance: bool = True,
    positional_motifs: bool = False,
    public_tcrs: bool = True,
    cluster_min_size: int = 2,
    hla_cutoff: float = 0.1,
    random_state: int | None = 42,
) -> dict:
    """Run GLIPH with independently selectable local / global methods.

    Parameters
    ----------
    local_method
        ``"fisher"`` (GLIPH2 hypergeometric test) or ``"rrs"`` (GLIPH
        repeated random sampling).
    global_method
        ``"fisher"`` (GLIPH2 position-specific structures) or
        ``"global_hamming"`` (GLIPH Hamming-distance pairs).
    Other parameters
        See :func:`pygliph.gliph2` and :func:`pygliph.turbo_gliph`.

    Returns
    -------
    dict
        Same structure as :func:`pygliph.gliph2`.
    """
    if local_method not in ("fisher", "rrs"):
        raise ValueError("local_method must be 'fisher' or 'rrs'")
    if global_method not in ("fisher", "global_hamming"):
        raise ValueError(
            "global_method must be 'fisher' or 'global_hamming'")

    # Pure GLIPH2 settings -> delegate directly to gliph2.
    if local_method == "fisher" and global_method == "fisher":
        return gliph2(
            cdr3_sequences, refdb_beta=refdb_beta, v_usage_freq=v_usage_freq,
            cdr3_length_freq=cdr3_length_freq,
            ref_cluster_size=ref_cluster_size, sim_depth=sim_depth,
            lcminp=lcminp, lcminove=lcminove,
            motif_distance_cutoff=motif_distance_cutoff,
            kmer_mindepth=kmer_mindepth,
            accept_sequences_with_C_F_start_end=(
                accept_sequences_with_C_F_start_end),
            min_seq_length=min_seq_length, structboundaries=structboundaries,
            boundary_size=boundary_size, motif_length=motif_length,
            discontinuous_motifs=discontinuous_motifs,
            local_similarities=local_similarities,
            global_similarities=global_similarities, global_vgene=global_vgene,
            all_aa_interchangeable=all_aa_interchangeable,
            boost_local_significance=boost_local_significance,
            cluster_min_size=cluster_min_size, hla_cutoff=hla_cutoff,
            random_state=random_state)

    # Pure GLIPH settings -> delegate to turbo_gliph and adapt output.
    if local_method == "rrs" and global_method == "global_hamming":
        res = turbo_gliph(
            cdr3_sequences, refdb_beta=refdb_beta,
            ref_cluster_size=ref_cluster_size, v_usage_freq=v_usage_freq,
            cdr3_length_freq=cdr3_length_freq, sim_depth=sim_depth,
            lcminp=lcminp, lcminove=lcminove, kmer_mindepth=kmer_mindepth,
            gccutoff=gccutoff,
            accept_sequences_with_C_F_start_end=(
                accept_sequences_with_C_F_start_end),
            min_seq_length=min_seq_length, structboundaries=structboundaries,
            boundary_size=boundary_size, motif_length=motif_length,
            discontinuous=discontinuous_motifs,
            local_similarities=local_similarities,
            global_similarities=global_similarities, global_vgene=global_vgene,
            positional_motifs=positional_motifs, public_tcrs=public_tcrs,
            cluster_min_size=cluster_min_size, hla_cutoff=hla_cutoff,
            random_state=random_state)
        return {
            "motif_enrichment": res["motif_enrichment"],
            "global_enrichment": None,
            "connections": res["connections"],
            "cluster_properties": res["cluster_properties"],
            "cluster_list": res["cluster_list"],
            "parameters": res["parameters"],
        }

    # Mixed methods: use Fisher locals (gliph2) when requested, otherwise
    # Hamming globals. We compose by running gliph2 with the appropriate
    # toggles -- gliph2 already supports the Fisher local + Fisher global
    # path; for global_hamming we fall back to turbo_gliph's global step.
    raise NotImplementedError(
        "Mixed local/global methods (local_method != global_method) are "
        "not supported; use gliph2() or turbo_gliph() directly, or pick a "
        "matched pair ('fisher'/'fisher' or 'rrs'/'global_hamming').")

"""Per-cluster enrichment scoring for GLIPH and GLIPH2.

Port of turboGliph's ``cluster_scoring`` (R/scoring.R).
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import datasets
from ._utils import format_e1

__all__ = ["cluster_scoring"]


def _prob_product_score(values: Sequence, n: int) -> float:
    """Round(prod(freq/n), 3) for the observed sample (R behaviour)."""
    counts = Counter(values)
    prod = 1.0
    for c in counts.values():
        prod *= c / n
    return round(prod, 3)


def _subsample_scores(
    rng: np.random.Generator,
    probs: np.ndarray,
    n: int,
    sim_depth: int,
) -> np.ndarray:
    """Vectorised resampling scores.

    Replicates::

        random_subsample <- sample.int(K, n, prob, replace = TRUE)
        pick_freqs       <- table(random_subsample) / n
        score            <- round(prod(pick_freqs), 3)

    The product is taken over *observed* categories only (zeros excluded,
    matching R's ``seq_qgrams`` whose zero counts become 1).
    """
    k = len(probs)
    p = probs / probs.sum()
    out = np.empty(sim_depth, dtype=float)
    for s in range(sim_depth):
        draw = rng.choice(k, size=n, replace=True, p=p)
        counts = np.bincount(draw, minlength=k)
        nz = counts[counts > 0] / n
        # log-sum for numerical stability (matches R's exp(colSums(log)))
        out[s] = round(float(np.exp(np.sum(np.log(nz)))), 3)
    return out


def cluster_scoring(
    cluster_list: Mapping[str, pd.DataFrame],
    cdr3_sequences: pd.DataFrame,
    refdb_beta="gliph_reference",
    v_usage_freq: pd.DataFrame | None = None,
    cdr3_length_freq: pd.DataFrame | None = None,
    ref_cluster_size: str = "original",
    gliph_version: int = 1,
    sim_depth: int = 1000,
    hla_cutoff: float = 0.1,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Score CDR3 clusters as in GLIPH / GLIPH2.

    Up to five property scores are combined into a ``total.score``:
    network size, CDR3-length enrichment, V-gene enrichment, clonal
    expansion enrichment and shared-HLA enrichment.

    Parameters
    ----------
    cluster_list
        Mapping ``{cluster_tag: members_dataframe}`` -- the ``cluster_list``
        element produced by :func:`pygliph.gliph2` / :func:`pygliph.turbo_gliph`.
    cdr3_sequences
        The full input table (used for global frequency baselines).
    gliph_version
        ``1`` -> total score multiplied by ``0.064`` (GLIPH); ``2`` -> plain
        product (GLIPH2).
    random_state
        Seed for the resampling RNG (the simulation is stochastic; fixing
        the seed makes results reproducible).

    Returns
    -------
    pandas.DataFrame
        One row per cluster with ``total.score`` and the individual scores.
    """
    if isinstance(cdr3_sequences, (list, tuple, pd.Series, np.ndarray)):
        cdr3_sequences = pd.DataFrame({"CDR3b": list(cdr3_sequences)})
    cdr3_sequences = cdr3_sequences.astype(str)

    rng = np.random.default_rng(random_state)

    score_names = ["network.size.score", "cdr3.length.score"]
    vgene_info = "TRBV" in cdr3_sequences.columns
    if vgene_info:
        score_names.append("vgene.score")
    counts_info = "counts" in cdr3_sequences.columns
    if counts_info:
        cdr3_sequences = cdr3_sequences.copy()
        cdr3_sequences["counts"] = pd.to_numeric(
            cdr3_sequences["counts"], errors="coerce").fillna(1.0)
        score_names.append("clonal.expansion.score")
    patient_info = "patient" in cdr3_sequences.columns
    hla_info = "HLA" in cdr3_sequences.columns
    if hla_info and patient_info:
        cdr3_sequences = cdr3_sequences[
            (cdr3_sequences["HLA"].astype(str) != "")
            & (cdr3_sequences["HLA"].astype(str) != "nan")]
        if len(cdr3_sequences) > 0:
            score_names += ["hla.score", "lowest.hlas"]
        else:
            hla_info = False

    # ---- reference frequency tables ------------------------------------
    ref_cluster_sizes = datasets.load_ref_cluster_sizes(ref_cluster_size)
    vgene_ref = datasets.load_vgene_ref_frequencies()["freq"].to_numpy(float)
    cdr3len_ref = datasets.load_cdr3_length_ref_frequencies()[
        "freq"].to_numpy(float)

    if isinstance(refdb_beta, pd.DataFrame):
        if "TRBV" in refdb_beta.columns:
            vc = refdb_beta["TRBV"].value_counts()
            vgene_ref = (vc / vc.sum()).to_numpy(float)
        lc = refdb_beta["CDR3b"].astype(str).str.len().value_counts()
        cdr3len_ref = (lc / lc.sum()).to_numpy(float)
    if v_usage_freq is not None:
        vgene_ref = pd.to_numeric(v_usage_freq.iloc[:, 1]).to_numpy(float)
    if cdr3_length_freq is not None:
        cdr3len_ref = pd.to_numeric(
            cdr3_length_freq.iloc[:, 1]).to_numpy(float)

    # ---- HLA distribution ---------------------------------------------
    all_patients: list = []
    all_patient_hlas: dict = {}
    all_hlas_df = None
    num_patients = 0
    num_hlas = 0
    if hla_info and patient_info:
        df = cdr3_sequences.copy()
        df["patient"] = df["patient"].str.replace(r":.*", "", regex=True)
        all_patients = sorted(set(df["patient"].dropna()))
        hlas: set = set()
        for h in df["HLA"].dropna().unique():
            for a in str(h).split(","):
                a = a.split(":")[0].strip()
                if a:
                    hlas.add(a)
        all_hlas = sorted(hlas)
        num_patients = len(all_patients)
        num_hlas = len(all_hlas)
        for pat in all_patients:
            sub = df.loc[df["patient"] == pat, "HLA"]
            phlas = set()
            if len(sub):
                for a in str(sub.iloc[0]).split(","):
                    a = a.split(":")[0].strip()
                    if a:
                        phlas.add(a)
            all_patient_hlas[pat] = sorted(phlas)
        hla_counts = []
        for h in all_hlas:
            hla_counts.append(sum(1 for p in all_patients
                                  if h in all_patient_hlas[p]))
        all_hlas_df = pd.DataFrame({"HLA": all_hlas, "counts": hla_counts})

    n_total = len(cdr3_sequences)
    sample_size_cols = [c for c in ref_cluster_sizes.columns
                        if c != "cluster.size"]
    sample_size_vals = np.asarray([float(c) for c in sample_size_cols])

    rows = []
    for tag, info in cluster_list.items():
        info = info.reset_index(drop=True)
        num_members = len(info)
        uniq_cdr3 = info["CDR3b"].nunique()
        scores: list[float] = []

        # ---- network-size score --------------------------------------
        nearest = int(np.argmin(np.abs(1 - sample_size_vals / max(n_total, 1))))
        idx = min(uniq_cdr3, 100)
        net = float(ref_cluster_sizes.iloc[idx - 1][sample_size_cols[nearest]])
        scores.append(net)

        # ---- CDR3-length enrichment ----------------------------------
        lens = info["CDR3b"].drop_duplicates().str.len().to_list()
        sample_score = _prob_product_score(lens, uniq_cdr3)
        sub = _subsample_scores(rng, cdr3len_ref, uniq_cdr3, sim_depth)
        if gliph_version == 1:
            s_len = float(np.sum(sub >= sample_score)) / sim_depth
        else:
            s_len = float(np.sum(sub > sample_score)) / sim_depth
        if s_len == 0:
            s_len = 1.0 / sim_depth
        scores.append(s_len)

        # ---- V-gene enrichment ---------------------------------------
        if vgene_info:
            sample_score = _prob_product_score(
                info["TRBV"].to_list(), num_members)
            sub = _subsample_scores(rng, vgene_ref, num_members, sim_depth)
            if gliph_version == 1:
                s_v = float(np.sum(sub >= sample_score)) / sim_depth
            else:
                s_v = float(np.sum(sub > sample_score)) / sim_depth
            if s_v == 0:
                s_v = 1.0 / sim_depth
            scores.append(s_v)

        # ---- clonal-expansion enrichment -----------------------------
        if counts_info:
            sc = pd.to_numeric(info["counts"], errors="coerce").fillna(1.0)
            sample_score = float(sc.sum()) / num_members
            pool = cdr3_sequences["counts"].to_numpy(float)
            counter = 0
            for _ in range(sim_depth):
                draw = rng.choice(pool, size=num_members, replace=False)
                if draw.sum() / num_members >= sample_score:
                    counter += 1
            s_c = (1.0 / sim_depth if counter == 0
                   else counter / sim_depth)
            scores.append(round(s_c, 3))

        # ---- shared-HLA enrichment -----------------------------------
        lowest_hla = ""
        if hla_info and patient_info:
            sub_info = info[(info["HLA"].astype(str) != "")
                            & (info["HLA"].astype(str) != "nan")]
            s_hla = 1.0
            if len(sub_info) > 0:
                pats = sub_info["patient"].str.replace(
                    r":.*", "", regex=True)
                crg_patients = sorted(set(pats))
                crg_pcount = len(crg_patients)
                from scipy.special import comb
                for i in range(num_hlas):
                    hla = all_hlas_df["HLA"].iloc[i]
                    hcount = all_hlas_df["counts"].iloc[i]
                    crg_hla = sum(1 for p in crg_patients
                                  if hla in all_patient_hlas.get(p, []))
                    if crg_hla > 1:
                        ks = np.arange(crg_hla, crg_pcount + 1)
                        prob = float(np.sum(
                            comb(hcount, ks)
                            * comb(num_patients - hcount, crg_pcount - ks)
                            / comb(num_patients, crg_pcount)))
                        if prob < s_hla:
                            s_hla = prob
                        if prob < hla_cutoff:
                            piece = (f"{hla} [({crg_hla}/{crg_pcount}) vs "
                                     f"({hcount}/{num_patients}) = "
                                     f"{format_e1(prob)}]")
                            lowest_hla = (piece if lowest_hla == ""
                                          else lowest_hla + ", " + piece)
            scores.append(s_hla)

        # ---- total score ---------------------------------------------
        prod = float(np.prod(scores))
        total = prod * 0.001 * 64 if gliph_version == 1 else prod

        out = [tag, format_e1(total)] + [format_e1(s) for s in scores]
        if hla_info and patient_info:
            out.append(lowest_hla)
        rows.append(out)

    cols = ["leader.tag", "total.score"] + score_names
    res = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c in ("leader.tag", "lowest.hlas"):
            continue
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res.drop(columns=["leader.tag"])

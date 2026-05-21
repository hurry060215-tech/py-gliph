"""De-novo CDR3 generation from a specificity group.

Port of turboGliph's ``de_novo_TCRs`` (R/de_novo_tcrs.R), following the
position-weight-matrix simulation of Glanville et al.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import datasets
from ._utils import as_numeric_format_e1

__all__ = ["de_novo_TCRs"]

# one-letter amino acids excluding B, J, O, U, X, Z (LETTERS[-c(2,10,15,21,24,26)])
AA_CODE = [c for c in "ACDEFGHIKLMNPQRSTVWY"]


def _pwm(seqs, n_pos):
    """Position weight matrix over ``n_pos`` N-terminal positions.

    Uses the 0.5% pseudocount rule from turboGliph.
    """
    pwm = np.zeros((n_pos, len(AA_CODE)), dtype=float)
    idx = {a: j for j, a in enumerate(AA_CODE)}
    for i in range(n_pos):
        freqs = np.zeros(len(AA_CODE))
        for s in seqs:
            if i < len(s) and s[i] in idx:
                freqs[idx[s[i]]] += 1
        zero = freqs == 0
        total = freqs.sum()
        if total > 0:
            freqs = freqs / total * (1 - zero.sum() * 0.005)
        freqs[zero] = 0.005
        pwm[i] = freqs
    return pd.DataFrame(pwm, columns=AA_CODE)


def _score(seqs, pwm: pd.DataFrame):
    """Product of position-specific amino-acid frequencies."""
    idx = {a: j for j, a in enumerate(AA_CODE)}
    mat = pwm.to_numpy()
    out = np.ones(len(seqs), dtype=float)
    for i in range(pwm.shape[0]):
        for k, s in enumerate(seqs):
            if i < len(s) and s[i] in idx:
                out[k] *= mat[i, idx[s[i]]]
            else:
                out[k] *= 0.0
    return out


def de_novo_TCRs(
    convergence_group_tag: str,
    clustering_output: dict,
    refdb_beta="gliph_reference",
    normalization: bool = False,
    accept_sequences_with_C_F_start_end: bool = True,
    sims: int = 100000,
    num_tops: int = 1000,
    min_length: int = 10,
    random_state: int | None = 42,
) -> dict:
    """Generate de-novo CDR3 sequences for one specificity group.

    Builds a position weight matrix from the group's members and simulates
    ``sims`` artificial CDR3s, returning the ``num_tops`` highest scoring.

    Parameters
    ----------
    convergence_group_tag
        Tag of the cluster (a key of ``clustering_output["cluster_list"]``).
    clustering_output
        Output dict of :func:`pygliph.gliph2` or :func:`pygliph.turbo_gliph`.
    normalization
        If ``True``, scores are normalised against the reference database.
    sims, num_tops, min_length
        Number of simulated sequences, number returned, and the count of
        N-terminal positions used for scoring.

    Returns
    -------
    dict
        Keys ``de_novo_sequences``, ``sample_sequences_scores``,
        ``cdr3_length_probability``, ``PWM_Scoring``, ``PWM_Prediction``.
    """
    rng = np.random.default_rng(random_state)
    crg = clustering_output["cluster_list"]
    if convergence_group_tag not in crg:
        raise ValueError(
            f"Convergence group '{convergence_group_tag}' not found.")
    members = crg[convergence_group_tag]
    all_seqs = members["CDR3b"].to_list()

    crg_seqs = [s for s in all_seqs if len(s) >= min_length]
    if not crg_seqs:
        raise ValueError(
            f"No group sequences reach min_length={min_length}.")

    crg_lens = [len(s) for s in crg_seqs]
    crg_n = len(crg_seqs)

    # ---- scoring PWM (min_length N-terminal positions) -----------------
    pwm_scoring = _pwm(crg_seqs, min_length)
    crg_scores = _score(crg_seqs, pwm_scoring)

    # ---- normalisation -------------------------------------------------
    crg_norm = np.zeros(crg_n)
    refseqs = None
    refseq_scores = None
    if normalization:
        if isinstance(refdb_beta, pd.DataFrame):
            ref = refdb_beta.copy().astype(str)
            cdr3 = ref.iloc[:, 0]
        else:
            ref = datasets.load_reference_db(refdb_beta)
            cdr3 = ref["CDR3b"]
        cand = cdr3.astype(str)
        if accept_sequences_with_C_F_start_end:
            cand = cand[cand.map(lambda s: bool(re.match(r"^C.*F$", s)))]
        cand = cand[cand.str.len() >= min_length]
        refseqs = cand.to_list()
        if refseqs:
            refseq_scores = _score(refseqs, pwm_scoring)
            for i in range(crg_n):
                crg_norm[i] = float(
                    np.sum(refseq_scores >= crg_scores[i])) / len(refseqs)
        else:
            normalization = False

    # ---- prediction PWMs per length -----------------------------------
    uniq_lens = list(dict.fromkeys(crg_lens))
    crg_len_prob = pd.DataFrame({
        "length": uniq_lens,
        "probability": [crg_lens.count(L) / crg_n for L in uniq_lens],
    })
    pwm_pred: dict[str, pd.DataFrame] = {}
    for L in uniq_lens:
        seqs_L = [s for s, ln in zip(crg_seqs, crg_lens) if ln == L]
        pwm = _pwm(seqs_L, L)
        if accept_sequences_with_C_F_start_end:
            pwm.iloc[0] = 0.0
            pwm.iloc[-1] = 0.0
            pwm.loc[0, "C"] = 1.0
            pwm.loc[L - 1, "F"] = 1.0
        pwm_pred[f"Length {L}"] = pwm

    # ---- simulate de-novo sequences -----------------------------------
    de_novo_lens = rng.choice(
        uniq_lens, size=sims, replace=True,
        p=crg_len_prob["probability"].to_numpy())
    de_novo = [""] * sims
    for L in uniq_lens:
        inds = np.where(de_novo_lens == L)[0]
        mat = pwm_pred[f"Length {L}"].to_numpy()
        for i in range(L):
            probs = mat[i]
            probs = probs / probs.sum()
            draw = rng.choice(len(AA_CODE), size=len(inds), p=probs)
            for k, idx in enumerate(inds):
                de_novo[idx] += AA_CODE[draw[k]]
    de_novo = list(dict.fromkeys(de_novo))
    if accept_sequences_with_C_F_start_end:
        de_novo = [s for s in de_novo if re.match(r"^C.*F$", s)]

    dn_scores = _score(de_novo, pwm_scoring)
    dn_scores = np.asarray([as_numeric_format_e1(x) for x in dn_scores])

    if normalization and refseq_scores is not None:
        dn_norm = np.asarray([
            as_numeric_format_e1(
                float(np.sum(refseq_scores >= dn_scores[i]))
                / len(refseqs))
            for i in range(len(de_novo))])
        order = np.argsort(dn_norm, kind="stable")
        de_novo = [de_novo[i] for i in order]
        dn_scores = dn_scores[order]
        dn_norm = dn_norm[order]
    else:
        order = np.argsort(-dn_scores, kind="stable")
        de_novo = [de_novo[i] for i in order]
        dn_scores = dn_scores[order]
        dn_norm = None

    num_tops = min(num_tops, len(de_novo))
    de_novo = de_novo[:num_tops]
    dn_scores = dn_scores[:num_tops]
    dn_lens = [len(s) for s in de_novo]
    if normalization and dn_norm is not None:
        dn_norm = dn_norm[:num_tops]
        de_novo_df = pd.DataFrame({"length": dn_lens, "seqs": de_novo,
                                   "norm_score": dn_norm, "score": dn_scores})
    else:
        de_novo_df = pd.DataFrame({"length": dn_lens, "seqs": de_novo,
                                   "score": dn_scores})

    # ---- ordered sample scores ----------------------------------------
    if normalization:
        sort_idx = np.argsort(crg_norm, kind="stable")
        sample_scores = pd.DataFrame({
            "seqs": [crg_seqs[i] for i in sort_idx],
            "norm_scores": crg_norm[sort_idx],
            "scores": crg_scores[sort_idx]})
    else:
        sort_idx = np.argsort(-crg_scores, kind="stable")
        sample_scores = pd.DataFrame({
            "seqs": [crg_seqs[i] for i in sort_idx],
            "scores": crg_scores[sort_idx]})

    return {
        "de_novo_sequences": de_novo_df,
        "sample_sequences_scores": sample_scores,
        "cdr3_length_probability": crg_len_prob,
        "PWM_Scoring": pwm_scoring,
        "PWM_Prediction": pwm_pred,
    }

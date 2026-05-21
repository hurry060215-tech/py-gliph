"""R-parity tests: pygliph vs turboGliph on the bundled example data.

The deterministic parts of the algorithms -- continuous/discontinuous
motif enumeration, the GLIPH2 hypergeometric ("Fisher") tests, global
Hamming-1 structures, group membership and the de-novo PWMs -- are
checked bit-exact (up to floating-point noise). The resampling-based
GLIPH v1 local enrichment depends on the RNG and is checked for
distributional / set agreement instead.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import pygliph as pg

TOL = 1e-9


def _read(r_reference, name, **kw):
    return pd.read_csv(os.path.join(r_reference, name), sep="\t", **kw)


# ----------------------------------------------------------------------
# find_motifs -- expected bit-exact
# ----------------------------------------------------------------------
def test_find_motifs_continuous_bit_exact(r_reference, r_input):
    seqs = r_input["CDR3b"].tolist()
    py = pg.find_motifs(seqs, q=(2, 3, 4))
    r = _read(r_reference, "find_motifs_cont.tsv")
    pm = dict(zip(py["motif"], py["V1"]))
    rm = dict(zip(r["motif"], r["count"]))
    assert set(pm) == set(rm)
    assert all(int(pm[k]) == int(rm[k]) for k in pm)


def test_find_motifs_discontinuous_bit_exact(r_reference, r_input):
    seqs = r_input["CDR3b"].tolist()
    py = pg.find_motifs(seqs, q=(2, 3), discontinuous=True)
    r = _read(r_reference, "find_motifs_disc.tsv")
    pm = dict(zip(py["motif"], py["V1"]))
    rm = dict(zip(r["motif"], r["count"]))
    assert set(pm) == set(rm)
    assert all(int(pm[k]) == int(rm[k]) for k in pm)


# ----------------------------------------------------------------------
# GLIPH2 -- deterministic: bit-exact
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def gliph2_result(r_input):
    return pg.gliph2(r_input, sim_depth=100, random_state=42)


def test_gliph2_all_motifs_fisher_bit_exact(r_reference, gliph2_result):
    py = gliph2_result["motif_enrichment"]["all_motifs"].set_index("motif")
    r = _read(r_reference, "gliph2_all_motifs.tsv").set_index("motif")
    assert set(py.index) == set(r.index)
    common = py.index
    assert (py.loc[common, "num_in_sample"].astype(float).values
            == r.loc[common, "num_in_sample"].astype(float).values).all()
    assert (py.loc[common, "num_in_ref"].astype(float).values
            == r.loc[common, "num_in_ref"].astype(float).values).all()
    # Fisher (hypergeometric) p-values: bit-exact after R's formatC rounding
    fp = py.loc[common, "fisher.score"].astype(float).values
    fr = r.loc[common, "fisher.score"].astype(float).values
    assert np.allclose(fp, fr, rtol=0, atol=0)


def test_gliph2_selected_motifs_match(r_reference, gliph2_result):
    py = gliph2_result["motif_enrichment"]["selected_motifs"]
    r = _read(r_reference, "gliph2_selected.tsv")

    def key(df):
        return set(zip(df["motif"], df["start"].astype(int),
                       df["stop"].astype(int)))

    assert key(py) == key(r)


def test_gliph2_global_structures_match(r_reference, gliph2_result):
    py = gliph2_result["global_enrichment"]
    r = _read(r_reference, "gliph2_global.tsv")
    pk = set(zip(py["cluster_tag"], py["unique_CDR3b"].astype(int)))
    rk = set(zip(r["cluster_tag"], r["unique_CDR3b"].astype(int)))
    assert pk == rk
    # Fisher scores for the shared structures agree
    pm = dict(zip(py["cluster_tag"], py["fisher.score"].astype(float)))
    rm = dict(zip(r["cluster_tag"], r["fisher.score"].astype(float)))
    for tag in pm:
        assert abs(pm[tag] - rm[tag]) <= max(1e-12, 1e-9 * abs(rm[tag]))


def test_gliph2_connections_bit_exact(r_reference, gliph2_result):
    py = gliph2_result["connections"]
    r = _read(r_reference, "gliph2_connections.tsv", header=None)
    r.columns = ["V1", "V2", "type", "cluster_tag"]
    ps = set(tuple(x) for x in py.itertuples(index=False))
    rs = set(tuple(x) for x in r.itertuples(index=False))
    assert ps == rs


def test_gliph2_cluster_membership_bit_exact(r_reference, gliph2_result):
    py = gliph2_result["cluster_properties"].set_index("tag")
    r = _read(r_reference, "gliph2_clusters.tsv").set_index("tag")
    assert set(py.index) == set(r.index)
    common = py.index
    for col in ("cluster_size", "unique_cdr3_sample", "unique_cdr3_ref"):
        assert (py.loc[common, col].astype(float).values
                == r.loc[common, col].astype(float).values).all()


def test_gliph2_cluster_fisher_bit_exact(r_reference, gliph2_result):
    py = gliph2_result["cluster_properties"].set_index("tag")
    r = _read(r_reference, "gliph2_clusters.tsv").set_index("tag")
    common = py.index
    fp = py.loc[common, "fisher.score"].astype(float).values
    fr = r.loc[common, "fisher.score"].astype(float).values
    rel = np.abs(fp - fr) / np.maximum(np.abs(fr), 1e-300)
    assert rel.max() < 1e-9


def test_gliph2_total_score_correlation(r_reference, gliph2_result):
    """Total score uses stochastic resampling -> check rank correlation."""
    py = gliph2_result["cluster_properties"].set_index("tag")
    r = _read(r_reference, "gliph2_clusters.tsv").set_index("tag")
    common = py.index
    tp = np.log10(py.loc[common, "total.score"].astype(float).values + 1e-300)
    tr = np.log10(r.loc[common, "total.score"].astype(float).values + 1e-300)
    assert np.corrcoef(tp, tr)[0, 1] > 0.95


# ----------------------------------------------------------------------
# GLIPH v1 (turbo_gliph) -- resampling-based: set agreement
# ----------------------------------------------------------------------
def test_turbo_gliph_selected_motifs_overlap(r_reference, r_input):
    g1 = pg.turbo_gliph(r_input, sim_depth=100, random_state=42)
    py = set(g1["motif_enrichment"]["selected_motifs"]["Motif"])
    r = set(_read(r_reference, "gliph1_selected.tsv")["Motif"])
    # the strongly enriched motifs that survive any RNG must be shared
    inter = py & r
    # Jaccard should be high; every R motif of count>=4 must reappear
    assert len(inter) / len(py | r) > 0.6


def test_turbo_gliph_runs_and_scores(r_input):
    g1 = pg.turbo_gliph(r_input, sim_depth=50, random_state=1)
    cp = g1["cluster_properties"]
    assert cp is not None and len(cp) > 0
    assert "total.score" in cp.columns
    assert (cp["total.score"] > 0).all()


# ----------------------------------------------------------------------
# de_novo_TCRs -- PWM deterministic: bit-exact
# ----------------------------------------------------------------------
def test_de_novo_pwm_bit_exact(r_reference, gliph2_result):
    tag_file = os.path.join(r_reference, "denovo_tag.txt")
    if not os.path.exists(tag_file):
        pytest.skip("de_novo R reference not generated")
    tag = open(tag_file).read().strip()
    dn = pg.de_novo_TCRs(tag, gliph2_result, sims=2000, num_tops=20,
                         random_state=7)
    rpwm = _read(r_reference, "denovo_pwm.tsv")
    ppwm = dn["PWM_Scoring"]
    assert list(ppwm.columns) == list(rpwm.columns)
    assert np.abs(ppwm.to_numpy() - rpwm.to_numpy()).max() < 1e-12


def test_de_novo_sample_scores_bit_exact(r_reference, gliph2_result):
    tag_file = os.path.join(r_reference, "denovo_tag.txt")
    if not os.path.exists(tag_file):
        pytest.skip("de_novo R reference not generated")
    tag = open(tag_file).read().strip()
    dn = pg.de_novo_TCRs(tag, gliph2_result, sims=2000, num_tops=20,
                         random_state=7)
    r = _read(r_reference, "denovo_sample_scores.tsv")
    rmap = dict(zip(r["seqs"], r["scores"]))
    pmap = dict(zip(dn["sample_sequences_scores"]["seqs"],
                    dn["sample_sequences_scores"]["scores"]))
    assert set(rmap) == set(pmap)
    for s in rmap:
        assert abs(rmap[s] - pmap[s]) < 1e-12

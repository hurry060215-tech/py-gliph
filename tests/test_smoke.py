"""Self-contained smoke tests for pygliph (no R required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pygliph as pg


# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------
def test_datasets_load():
    df = pg.datasets.load_gliph_input_data()
    assert {"CDR3b", "TRBV", "patient", "counts", "HLA"} <= set(df.columns)
    assert len(df) > 1000

    ref = pg.datasets.load_reference_db("gliph_reference")
    assert "CDR3b" in ref.columns
    assert len(ref) > 100000

    assert len(pg.datasets.load_blosum_vec()) > 0
    assert set(pg.datasets.load_gtrb()) == {"gTRBV", "gTRBD", "gTRBJ"}
    assert pg.datasets.load_ref_cluster_sizes("original").shape[0] == 100
    assert pg.datasets.load_vgene_ref_frequencies().shape[1] == 2
    assert pg.datasets.load_cdr3_length_ref_frequencies().shape[1] == 2


def test_datasets_reject_unknown_reference():
    with pytest.raises(ValueError):
        pg.datasets.load_reference_db("not_a_db")


# ----------------------------------------------------------------------
# find_motifs
# ----------------------------------------------------------------------
def test_find_motifs_counts():
    seqs = ["CASSLAPGATF", "CASSLAPGQTF", "CDEFGH"]
    fm = pg.find_motifs(seqs, q=(2, 3))
    counts = dict(zip(fm["motif"], fm["V1"]))
    # "AS" occurs once in each of the first two sequences
    assert counts["AS"] == 2
    # "LAP" 3-mer occurs once per first two sequences
    assert counts["LAP"] == 2
    # a motif unique to seq 3
    assert counts["EFG"] == 1


def test_find_motifs_kmer_mindepth():
    seqs = ["AAAA", "AAAA"]
    fm = pg.find_motifs(seqs, q=(2,), kmer_mindepth=4)
    # "AA" occurs 3x per seq -> 6 total; survives mindepth 4
    assert dict(zip(fm["motif"], fm["V1"])) == {"AA": 6}


def test_find_motifs_discontinuous():
    seqs = ["CABCD", "CAXCD"]
    fm = pg.find_motifs(seqs, q=(3,), discontinuous=True)
    motifs = set(fm["motif"])
    # discontinuous 3-mer with a wildcard middle position
    assert any("." in m for m in motifs)


# ----------------------------------------------------------------------
# gliph2
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_input():
    return pg.datasets.load_gliph_input_data().iloc[:200].copy()


def test_gliph2_structure(small_input):
    res = pg.gliph2(small_input, sim_depth=30, random_state=0)
    for key in ("motif_enrichment", "global_enrichment", "connections",
                "cluster_properties", "cluster_list", "parameters"):
        assert key in res
    assert res["parameters"]["gliph_version"] == 2
    me = res["motif_enrichment"]
    assert "selected_motifs" in me and "all_motifs" in me
    assert len(res["connections"]) > 0


def test_gliph2_vector_input():
    seqs = pg.datasets.load_gliph_input_data()["CDR3b"].iloc[:150].tolist()
    res = pg.gliph2(seqs, sim_depth=20, random_state=0)
    assert res["cluster_properties"] is not None


def test_gliph2_local_only(small_input):
    res = pg.gliph2(small_input, sim_depth=20, global_similarities=False,
                    random_state=0)
    assert res["parameters"]["global_similarities"] is False


def test_gliph2_global_only(small_input):
    res = pg.gliph2(small_input, sim_depth=20, local_similarities=False,
                    random_state=0)
    assert res["parameters"]["local_similarities"] is False


def test_gliph2_deterministic_motifs(small_input):
    a = pg.gliph2(small_input, sim_depth=20, random_state=0)
    b = pg.gliph2(small_input, sim_depth=20, random_state=0)
    am = a["motif_enrichment"]["all_motifs"].sort_values("motif")
    bm = b["motif_enrichment"]["all_motifs"].sort_values("motif")
    pd.testing.assert_frame_equal(am.reset_index(drop=True),
                                  bm.reset_index(drop=True))


# ----------------------------------------------------------------------
# turbo_gliph
# ----------------------------------------------------------------------
def test_turbo_gliph_structure(small_input):
    res = pg.turbo_gliph(small_input, sim_depth=30, random_state=0)
    for key in ("sample_log", "motif_enrichment", "connections",
                "cluster_properties", "cluster_list", "parameters"):
        assert key in res
    assert res["parameters"]["gliph_version"] == 1


def test_turbo_gliph_gccutoff(small_input):
    res = pg.turbo_gliph(small_input, sim_depth=20, gccutoff=2,
                         random_state=0)
    assert res["parameters"]["gccutoff"] == 2


# ----------------------------------------------------------------------
# gliph_combined
# ----------------------------------------------------------------------
def test_gliph_combined_fisher(small_input):
    res = pg.gliph_combined(small_input, local_method="fisher",
                            global_method="fisher", sim_depth=20,
                            random_state=0)
    assert res["parameters"]["gliph_version"] == 2


def test_gliph_combined_rrs(small_input):
    res = pg.gliph_combined(small_input, local_method="rrs",
                            global_method="global_hamming", sim_depth=20,
                            random_state=0)
    assert res["parameters"]["gliph_version"] == 1


def test_gliph_combined_rejects_mixed(small_input):
    with pytest.raises(NotImplementedError):
        pg.gliph_combined(small_input, local_method="fisher",
                          global_method="global_hamming")


# ----------------------------------------------------------------------
# cluster_scoring
# ----------------------------------------------------------------------
def test_cluster_scoring_standalone(small_input):
    res = pg.gliph2(small_input, sim_depth=20, random_state=0)
    sc = pg.cluster_scoring(res["cluster_list"], small_input,
                            gliph_version=2, sim_depth=20, random_state=0)
    assert "total.score" in sc.columns
    assert "network.size.score" in sc.columns
    assert len(sc) == len(res["cluster_list"])
    assert (sc["total.score"] > 0).all()


# ----------------------------------------------------------------------
# de_novo_TCRs
# ----------------------------------------------------------------------
def test_de_novo_tcrs(small_input):
    res = pg.gliph2(small_input, sim_depth=20, random_state=0)
    tag = None
    for t, cl in res["cluster_list"].items():
        if (cl["CDR3b"].str.len() >= 10).sum() >= 3:
            tag = t
            break
    assert tag is not None
    dn = pg.de_novo_TCRs(tag, res, sims=1000, num_tops=20, random_state=0)
    assert "de_novo_sequences" in dn
    assert "PWM_Scoring" in dn
    assert len(dn["de_novo_sequences"]) <= 20
    # PWM rows sum to ~1 with pseudocounts
    row_sums = dn["PWM_Scoring"].sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# io round-trip
# ----------------------------------------------------------------------
def test_io_round_trip(small_input, tmp_path):
    res = pg.gliph2(small_input, sim_depth=20, random_state=0)
    folder = str(tmp_path / "gliph_out")
    pg.save_gliph_output(res, folder)
    loaded = pg.load_gliph_output(folder)
    assert loaded["parameters"]["gliph_version"] == 2
    assert set(loaded["cluster_list"]) == set(res["cluster_list"])
    assert len(loaded["cluster_properties"]) == len(res["cluster_properties"])


# ----------------------------------------------------------------------
# plot_network
# ----------------------------------------------------------------------
def test_plot_network(small_input):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    res = pg.gliph2(small_input, sim_depth=20, random_state=0)
    ax = pg.plot_network(res)
    assert ax is not None

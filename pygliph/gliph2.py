"""GLIPH2 -- Grouping of Lymphocyte Interactions by Paratope Hotspots, v2.

Port of turboGliph's ``gliph2`` (R/gliph2.R), following Huang et al.,
Nat. Biotechnol. 2020. Local (motif) similarity significance is assessed
with Fisher's exact test against a reference TCR pool; global similarity
groups CDR3s of equal length differing at one position.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from . import datasets
from ._utils import as_numeric_format_e1, format_e1
from .motifs import find_motifs
from .scoring import cluster_scoring

__all__ = ["gliph2"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _phyper_greater(q, m, n, k) -> np.ndarray:
    """R ``phyper(q, m, n, k, lower.tail = FALSE)``.

    ``q`` = white drawn - 1, ``m`` = white, ``n`` = black, ``k`` = drawn.
    Returns ``P(X > q)`` = survival function of the hypergeometric.
    """
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)
    n = np.asarray(n, dtype=float)
    return hypergeom.sf(q, m + n, m, k)


def _str_locate_all(seqs: Sequence[str], pattern: str):
    """All 1-based start positions of ``pattern`` in each sequence."""
    out = []
    plen = len(pattern)
    for s in seqs:
        starts = []
        start = 0
        while True:
            idx = s.find(pattern, start)
            if idx < 0:
                break
            starts.append(idx + 1)
            start = idx + 1
        out.append(starts)
    return out


def _prepare_sequences(cdr3_sequences, global_vgene):
    """Normalise input into a DataFrame with a ``seq_ID`` column."""
    if isinstance(cdr3_sequences, (list, tuple, pd.Series, np.ndarray)):
        seq = pd.DataFrame({"CDR3b": [str(x) for x in cdr3_sequences]})
        if global_vgene:
            raise ValueError(
                "global_vgene=True requires a 'TRBV' column in cdr3_sequences")
    elif isinstance(cdr3_sequences, pd.DataFrame):
        df = cdr3_sequences.copy()
        if df.shape[1] == 1:
            seq = pd.DataFrame({"CDR3b": df.iloc[:, 0].astype(str)})
        else:
            if "CDR3b" not in df.columns:
                raise ValueError("cdr3_sequences needs a 'CDR3b' column")
            seq = pd.DataFrame({"CDR3b": df["CDR3b"].astype(str)})
        for col in ("TRBV", "patient", "HLA", "counts"):
            if col in df.columns:
                seq[col] = df[col]
        if global_vgene and "TRBV" not in seq.columns:
            raise ValueError(
                "global_vgene=True requires a 'TRBV' column in cdr3_sequences")
        extra = [c for c in df.columns if c not in seq.columns]
        for c in extra:
            seq[c] = df[c]
        seq = seq.astype(str)
    else:
        raise TypeError("cdr3_sequences must be a vector or DataFrame")

    valid = seq["CDR3b"].map(
        lambda s: bool(re.match(r"^[ACDEFGHIKLMNPQRSTVWY]*$", s)))
    seq = seq[valid].reset_index(drop=True)
    if len(seq) == 0:
        raise ValueError("No valid CDR3b amino-acid sequences found.")
    seq.insert(0, "seq_ID", np.arange(1, len(seq) + 1))
    return seq


def _identify_n_regions(sequences: pd.DataFrame):
    """Find non-germline (N/P) encoded sub-regions of each CDR3.

    Returns a DataFrame with columns ``start_1, stop_1, start_2, stop_2``
    (1-based inclusive ranges; 0 means absent), reproducing the gTRB
    walk in turboGliph's ``gliph2`` "boost_local_significance" block.
    """
    gtrb = datasets.load_gtrb()
    gv = gtrb["gTRBV"]
    gj = gtrb["gTRBJ"]
    gd = gtrb["gTRBD"]
    cdr3 = sequences["CDR3b"].to_list()
    n = len(cdr3)

    start_1 = np.zeros(n, dtype=int)
    stop_1 = np.zeros(n, dtype=int)
    start_2 = np.zeros(n, dtype=int)
    stop_2 = np.zeros(n, dtype=int)

    # ---- V-gene N-terminal match ---------------------------------------
    gv_by_len = {}
    for s, ln in zip(gv["seq"], gv["len"]):
        gv_by_len.setdefault(int(ln), set()).add(s)
    remaining = list(range(n))
    for i in range(int(gv["len"].max()), int(gv["len"].min()) - 1, -1):
        frags = gv_by_len.get(i, set())
        still = []
        for idx in remaining:
            if cdr3[idx][:i] in frags:
                start_1[idx] = i + 1
            else:
                still.append(idx)
        remaining = still
        if not remaining:
            break

    # ---- J-gene C-terminal match ---------------------------------------
    gj_by_len = {}
    for s, ln in zip(gj["seq"], gj["len"]):
        gj_by_len.setdefault(int(ln), set()).add(s)
    remaining = list(range(n))
    for i in range(int(gj["len"].max()), int(gj["len"].min()) - 1, -1):
        frags = gj_by_len.get(i, set())
        still = []
        for idx in remaining:
            s = cdr3[idx]
            if s[len(s) - i:] in frags:
                stop_2[idx] = i
            else:
                still.append(idx)
        remaining = still
        if not remaining:
            break
    seq_len = np.asarray([len(s) for s in cdr3])
    stop_2 = seq_len - stop_2

    # ---- D-gene internal match -----------------------------------------
    gd_by_len = {}
    for s, ln in zip(gd["seq"], gd["len"]):
        gd_by_len.setdefault(int(ln), []).append(s)
    remaining = list(range(n))
    act_sub = [cdr3[idx][start_1[idx] - 1:stop_2[idx]] if start_1[idx] >= 1
               else cdr3[idx][0:stop_2[idx]] for idx in remaining]
    for i in range(int(gd["len"].max()), int(gd["len"].min()) - 1, -1):
        if not remaining:
            break
        maxlen = max((len(s) for s in act_sub), default=0)
        if maxlen - i + 1 < 1:
            continue
        for frag in gd_by_len.get(i, []):
            new_remaining, new_sub = [], []
            for idx, sub in zip(remaining, act_sub):
                # most N-terminal occurrence of frag (length i)
                pos = -1
                for j in range(len(sub) - i + 1):
                    if sub[j:j + i] == frag:
                        pos = j + 1  # 1-based
                        break
                if pos == 1:
                    start_1[idx] = start_1[idx] + i
                elif pos > 1:
                    stop_1[idx] = start_1[idx] + pos - 2
                    start_2[idx] = stop_1[idx] + i + 1
                if pos > 0:
                    continue  # matched -> drop
                new_remaining.append(idx)
                new_sub.append(sub)
            remaining, act_sub = new_remaining, new_sub
            if not remaining:
                break

    return pd.DataFrame({"start_1": start_1, "stop_1": stop_1,
                         "start_2": start_2, "stop_2": stop_2})


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------
def gliph2(
    cdr3_sequences,
    refdb_beta="gliph_reference",
    v_usage_freq: pd.DataFrame | None = None,
    cdr3_length_freq: pd.DataFrame | None = None,
    ref_cluster_size: str = "original",
    sim_depth: int = 1000,
    lcminp: float = 0.01,
    lcminove=(1000, 100, 10),
    motif_distance_cutoff: int = 3,
    kmer_mindepth: int = 3,
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
    cluster_min_size: int = 2,
    hla_cutoff: float = 0.1,
    random_state: int | None = 42,
) -> dict:
    """Identify TCR specificity groups with the GLIPH2 algorithm.

    Parameters
    ----------
    cdr3_sequences
        A sequence of CDR3b strings, or a DataFrame with a ``CDR3b``
        column and optional ``TRBV``, ``patient``, ``HLA``, ``counts``.
    refdb_beta
        ``"gliph_reference"`` or a DataFrame ``[CDR3b, TRBV]`` reference.
    lcminp, lcminove, kmer_mindepth
        Local-motif filters: max Fisher p-value, min fold enrichment
        (per motif length) and min sample occurrences.
    motif_distance_cutoff
        Motif positions in one local cluster may differ by less than this.
    structboundaries, boundary_size
        Trim ``boundary_size`` residues at each terminus before analysis.
    global_vgene
        Restrict global similarity to TCRs sharing a V-gene.
    all_aa_interchangeable
        If ``False``, global edges require a BLOSUM62 >= 0 substitution.
    boost_local_significance
        Halve a local cluster's Fisher p per unique CDR3 whose motif
        overlaps a non-germline (N/P) encoded region.

    Returns
    -------
    dict
        Keys ``motif_enrichment``, ``global_enrichment``, ``connections``,
        ``cluster_properties``, ``cluster_list``, ``parameters`` -- the
        same structure as turboGliph's ``gliph2``.
    """
    lcminove = list(lcminove)
    motif_length = list(motif_length)
    motif_diffs = max(0, motif_distance_cutoff - 1)
    if structboundaries:
        min_seq_length = max(min_seq_length, 2 * boundary_size + 1)

    sequences = _prepare_sequences(cdr3_sequences, global_vgene)
    vgene_info = "TRBV" in sequences.columns

    # ---- sample motif region -------------------------------------------
    seqs_all = sequences["CDR3b"].to_list()
    if accept_sequences_with_C_F_start_end:
        seqs = [s for s in seqs_all if re.match(r"^C.*F$", s)]
    else:
        seqs = list(seqs_all)
    seqs = list(dict.fromkeys(seqs))  # unique, order-preserving
    seqs = [s for s in seqs if len(s) >= min_seq_length]
    if not seqs:
        raise ValueError("No CDR3b sequences passed the length/C..F filter.")
    if structboundaries:
        motif_region = [s[boundary_size:len(s) - boundary_size] for s in seqs]
    else:
        motif_region = list(seqs)

    # ---- reference database --------------------------------------------
    if isinstance(refdb_beta, pd.DataFrame):
        ref = refdb_beta.copy().astype(str)
        if "CDR3b" not in ref.columns:
            ref = ref.rename(columns={ref.columns[0]: "CDR3b"})
        if "TRBV" not in ref.columns:
            ref["TRBV"] = "" if not global_vgene else ref.iloc[:, 1]
        ref = ref[["CDR3b", "TRBV"]]
    else:
        ref = datasets.load_reference_db(refdb_beta)[["CDR3b", "TRBV"]].copy()
    if accept_sequences_with_C_F_start_end:
        ref = ref[ref["CDR3b"].map(lambda s: bool(re.match(r"^C.*F$", s)))]
    ref = ref.drop_duplicates()
    ref = ref[ref["CDR3b"].str.len() >= min_seq_length]
    ref = ref[ref["CDR3b"].map(
        lambda s: bool(re.match(r"^[ACDEFGHIKLMNPQRSTVWY]*$", s)))]
    ref = ref.reset_index(drop=True)

    ref_unique_cdr3 = ref["CDR3b"].drop_duplicates().to_list()
    if structboundaries:
        ref_motif_region = [s[boundary_size:len(s) - boundary_size]
                            for s in ref_unique_cdr3]
    else:
        ref_motif_region = list(ref_unique_cdr3)

    n_ref = len(ref_unique_cdr3)
    n_sample = len(seqs)

    # ====================================================================
    # Part 1: Local similarities
    # ====================================================================
    part_1_res = None
    part_1_all = None
    if local_similarities:
        ref_motifs = find_motifs(ref_motif_region, q=motif_length,
                                 discontinuous=discontinuous_motifs)
        ref_counts = (ref_motifs.groupby("motif", sort=False)["V1"].sum()
                      if len(ref_motifs) else pd.Series(dtype=float))

        smp_motifs = find_motifs(motif_region, q=motif_length,
                                 discontinuous=discontinuous_motifs)
        smp_counts = (smp_motifs.groupby("motif", sort=False)["V1"].sum()
                      if len(smp_motifs) else pd.Series(dtype=float))

        motifs_df = pd.DataFrame({
            "motif": smp_counts.index,
            "num_in_sample": smp_counts.to_numpy(float),
        })
        motifs_df["num_in_ref"] = motifs_df["motif"].map(
            ref_counts).fillna(0.0).to_numpy(float)

        # Fisher (hypergeometric) significance
        fp = _phyper_greater(
            motifs_df["num_in_sample"].to_numpy() - 1,
            motifs_df["num_in_ref"].to_numpy()
            + motifs_df["num_in_sample"].to_numpy(),
            n_ref + n_sample - motifs_df["num_in_ref"].to_numpy()
            - motifs_df["num_in_sample"].to_numpy(),
            n_sample)
        motifs_df["fisher.score"] = [as_numeric_format_e1(x) for x in fp]
        motifs_df["num_fold"] = np.round(
            motifs_df["num_in_sample"].to_numpy()
            / (motifs_df["num_in_ref"].to_numpy() + 0.01)
            / n_sample * n_ref, 1)

        # per-motif minimum fold change
        if len(lcminove) == 1:
            temp_minove = np.full(len(motifs_df), lcminove[0], dtype=float)
        else:
            nam_len = motifs_df["motif"].str.len().to_numpy()
            has_dot = motifs_df["motif"].str.contains(".", regex=False)
            nam_len = nam_len - has_dot.to_numpy().astype(int)
            temp_minove = np.zeros(len(motifs_df), dtype=float)
            for ml, ov in zip(motif_length, lcminove):
                temp_minove[nam_len == ml] = ov

        sel_mask = ((motifs_df["num_fold"].to_numpy() >= temp_minove)
                    & (motifs_df["fisher.score"].to_numpy() <= lcminp)
                    & (motifs_df["num_in_sample"].to_numpy() >= kmer_mindepth))
        sel = motifs_df[sel_mask].reset_index(drop=True)

        part_1_all = motifs_df.copy()
        if len(sel) > 0:
            # position-restricted preclustering
            recs = []
            for i in range(len(sel)):
                motif = sel["motif"].iloc[i]
                all_pos = _str_locate_all(motif_region, motif)
                id_starts = [(idx, st) for idx, sts in enumerate(all_pos)
                             for st in sts]
                if not id_starts:
                    continue
                starts = sorted({st for _, st in id_starts})
                rotated = starts[1:] + [starts[-1]]
                breaks = [j for j in range(len(starts))
                          if rotated[j] - starts[j] > motif_diffs]
                if motif_distance_cutoff < 1 or not breaks:
                    ids = sorted({idx for idx, _ in id_starts})
                    members = sorted({seqs[idx] for idx in ids})
                    recs.append((i, 0, 0, len(members), " ".join(members)))
                else:
                    breaks = sorted(set(breaks + [len(starts) - 1]))
                    min_start = min(s for _, s in id_starts)
                    for j in breaks:
                        hi = starts[j]
                        ids = sorted({idx for idx, st in id_starts
                                      if min_start <= st <= hi})
                        members = sorted({seqs[idx] for idx in ids})
                        recs.append((i, min_start, hi, len(members),
                                     " ".join(members)))
                        min_start = rotated[j]

            sel_pos = pd.DataFrame(
                recs, columns=["motifID", "start", "stop", "counts",
                               "members"])
            clusters = sel.iloc[sel_pos["motifID"].to_numpy()].reset_index(
                drop=True)
            clusters["start"] = sel_pos["start"].to_numpy()
            clusters["stop"] = sel_pos["stop"].to_numpy()
            clusters["num_in_sample"] = sel_pos["counts"].to_numpy()
            clusters["members"] = sel_pos["members"].to_numpy()
            clusters.loc[clusters["start"] == 0, "start"] = 1
            max_mr = max((len(m) for m in motif_region), default=1)
            clusters.loc[clusters["stop"] == 0, "stop"] = max_mr
            add = (1 if structboundaries else 0) * boundary_size
            clusters["start"] = clusters["start"] + add
            clusters["stop"] = clusters["stop"] + add

            fp2 = _phyper_greater(
                clusters["num_in_sample"].to_numpy() - 1,
                clusters["num_in_ref"].to_numpy()
                + clusters["num_in_sample"].to_numpy(),
                n_ref + n_sample - clusters["num_in_ref"].to_numpy()
                - clusters["num_in_sample"].to_numpy(),
                n_sample)
            clusters["fisher.score"] = [as_numeric_format_e1(x) for x in fp2]
            part_1_res = clusters
        else:
            part_1_res = pd.DataFrame(columns=[
                "motif", "num_in_sample", "num_in_ref", "fisher.score",
                "num_fold", "start", "stop", "members"])
            local_similarities = False
    else:
        part_1_res = None
        part_1_all = None

    # ====================================================================
    # Part 2: Global similarities
    # ====================================================================
    blosum = datasets.load_blosum_vec()
    part_2_res = None
    if global_similarities:
        sample_seqs = [s for s in seqs_all
                       if (not accept_sequences_with_C_F_start_end
                           or re.match(r"^C.*F$", s))]
        sample_seqs = list(dict.fromkeys(sample_seqs))
        sample_seqs = [s for s in sample_seqs if len(s) >= min_seq_length]
        struct_of = {}
        for s in sample_seqs:
            struct_of[s] = (s[boundary_size:len(s) - boundary_size]
                            if structboundaries else s)

        # expand: one variable position per struct, keep tags occurring >=2
        exp = []  # (seq, struct, pos, aa, tag)
        max_nchar = max((len(struct_of[s]) for s in sample_seqs), default=0)
        for pos in range(1, max_nchar + 1):
            tags = []
            for s in sample_seqs:
                st = struct_of[s]
                if len(st) >= pos:
                    aa = st[pos - 1]
                    tag = st[:pos - 1] + "%" + st[pos:]
                    tags.append((s, st, pos, aa, tag))
            tag_counter = Counter(t[4] for t in tags)
            for rec in tags:
                if tag_counter[rec[4]] >= 2:
                    exp.append(rec)

        unq_sample_struct = set(r[4] for r in exp)
        cluster_list_global = None
        if unq_sample_struct:
            # reference structs occurring in the sample set
            ref_struct_of = {}
            for s in ref_unique_cdr3:
                ref_struct_of[s] = (s[boundary_size:len(s) - boundary_size]
                                    if structboundaries else s)
            ref_max = max((len(v) for v in ref_struct_of.values()), default=0)
            ref_tag_counter: Counter = Counter()
            for pos in range(1, min(ref_max, max_nchar) + 1):
                for s in ref_unique_cdr3:
                    st = ref_struct_of[s]
                    if len(st) >= pos:
                        tag = st[:pos - 1] + "%" + st[pos:]
                        if tag in unq_sample_struct:
                            ref_tag_counter[tag] += 1

            sample_tag_counter = Counter(r[4] for r in exp)
            # vgene per sequence
            vg_of = {}
            if global_vgene:
                for _, row in sequences.iterrows():
                    vg_of[row["CDR3b"]] = row["TRBV"]

            # build edges within each struct
            edges = []
            by_tag: dict[str, list] = {}
            for rec in exp:
                by_tag.setdefault(rec[4], []).append(rec)
            for tag, members in by_tag.items():
                members = list(dict.fromkeys(
                    (m[0], m[4], m[3], m[2],
                     vg_of.get(m[0], " ") if global_vgene else " ")
                    for m in members))
                if len(members) < 2:
                    continue
                for a, b in combinations(members, 2):
                    keep = True
                    if global_vgene and a[4] != b[4]:
                        keep = False
                    if not all_aa_interchangeable and (a[2] + b[2]) \
                            not in blosum:
                        keep = False
                    if keep:
                        edges.append((
                            "#".join(str(x) for x in a),
                            "#".join(str(x) for x in b)))

            if edges:
                import networkx as nx
                g = nx.Graph()
                g.add_edges_from(edges)
                comps = list(nx.connected_components(g))
                rows = []
                for comp in comps:
                    parsed = [node.split("#") for node in comp]
                    csize = len(set(p[0] for p in parsed))  # unique CDR3
                    cdr3s = list(dict.fromkeys(p[0] for p in parsed))
                    aas = "".join(sorted(set(p[2] for p in parsed)))
                    tag = parsed[0][1]
                    trbv = parsed[0][4]
                    num_ref = ref_tag_counter.get(tag, 0)
                    rows.append((tag, len(comp), csize, num_ref, aas, trbv,
                                 " ".join(cdr3s)))
                cluster_list_global = pd.DataFrame(rows, columns=[
                    "cluster_tag", "cluster_size", "unique_CDR3b",
                    "num_in_ref", "aa_at_position", "TRBV", "CDR3b"])
                fp = _phyper_greater(
                    cluster_list_global["unique_CDR3b"].to_numpy() - 1,
                    cluster_list_global["num_in_ref"].to_numpy()
                    + cluster_list_global["unique_CDR3b"].to_numpy(),
                    n_ref + len(sample_seqs)
                    - cluster_list_global["num_in_ref"].to_numpy()
                    - cluster_list_global["unique_CDR3b"].to_numpy(),
                    len(sample_seqs))
                cluster_list_global["fisher.score"] = fp
                cluster_list_global = cluster_list_global[[
                    "cluster_tag", "cluster_size", "unique_CDR3b",
                    "num_in_ref", "fisher.score", "aa_at_position", "TRBV",
                    "CDR3b"]]
        if cluster_list_global is None:
            cluster_list_global = pd.DataFrame(columns=[
                "cluster_tag", "cluster_size", "unique_CDR3b", "num_in_ref",
                "fisher.score", "aa_at_position", "TRBV", "CDR3b"])
            global_similarities = False
        part_2_res = cluster_list_global

    # ====================================================================
    # Part 3: Clustering
    # ====================================================================
    range_df = None
    if boost_local_significance:
        range_df = _identify_n_regions(sequences)

    merged = []
    if local_similarities and part_1_res is not None and len(part_1_res) > 0:
        for i in range(len(part_1_res)):
            r = part_1_res.iloc[i]
            merged.append({
                "type": "local",
                "tag": f"{r['motif']}_{int(r['start'])}_{int(r['stop'])}",
                "cluster_size": 0,
                "unique_cdr3_sample": r["num_in_sample"],
                "unique_cdr3_ref": r["num_in_ref"],
                "OvE": r["num_fold"],
                "fisher.score": r["fisher.score"],
                "members": r["members"],
                "_motif": r["motif"], "_start": int(r["start"]),
                "_stop": int(r["stop"]),
            })
    if global_similarities and part_2_res is not None and len(part_2_res) > 0:
        for i in range(len(part_2_res)):
            r = part_2_res.iloc[i]
            if global_vgene:
                tag = f"{r['cluster_tag']}_{r['TRBV']}_{r['aa_at_position']}"
            else:
                tag = f"{r['cluster_tag']}_{r['aa_at_position']}"
            merged.append({
                "type": "global", "tag": tag,
                "cluster_size": r["cluster_size"],
                "unique_cdr3_sample": r["unique_CDR3b"],
                "unique_cdr3_ref": r["num_in_ref"],
                "OvE": 0,
                "fisher.score": r["fisher.score"],
                "members": r["CDR3b"], "_trbv": r["TRBV"],
            })

    cluster_list: dict[str, pd.DataFrame] = {}
    clone_network_rows = []
    merged_kept = []
    if merged:
        # build cluster member tables.  R selects rows via
        # ``sequences[sequences$CDR3b %in% act_seqs, ]`` which keeps the
        # *input row order* -- reproduce that here.
        for m in merged:
            act = set(m["members"].split(" ")) if m["members"] else set()
            mask = sequences["CDR3b"].isin(act)
            if m["type"] == "global" and global_vgene:
                mask = mask & (sequences["TRBV"] == m["_trbv"])
            m["_table"] = sequences[mask].reset_index(drop=True)

        # update local cluster size and boost significance
        for m in merged:
            if m["type"] == "local":
                m["cluster_size"] = m["_table"]["CDR3b"].nunique()
                if boost_local_significance and range_df is not None:
                    start_motif = m["_start"]
                    stop_motif = m["_stop"] + len(m["_motif"]) - 1
                    ids = m["_table"]["seq_ID"].astype(int).to_numpy() - 1
                    overlap = (
                        ((range_df["start_1"].to_numpy()[ids] <= stop_motif)
                         & (range_df["stop_1"].to_numpy()[ids] >= start_motif))
                        | ((range_df["start_2"].to_numpy()[ids] <= stop_motif)
                           & (range_df["stop_2"].to_numpy()[ids]
                              >= start_motif)))
                    m["fisher.score"] = m["fisher.score"] / (
                        2 ** int(np.sum(overlap)))

        # drop clusters below cluster_min_size
        for m in merged:
            if m["cluster_size"] >= cluster_min_size:
                merged_kept.append(m)
                cluster_list[m["tag"]] = m["_table"]

        # build clone network
        for m in merged_kept:
            members = m["_table"]["CDR3b"].to_list()
            if m["type"] == "local":
                motif = m["tag"].split("_")[0]
                frags = [(c[boundary_size:len(c) - boundary_size]
                          if structboundaries else c) for c in members]
                pos = [f.find(motif) for f in frags]
                for (ia, a), (ib, b) in combinations(
                        list(enumerate(members)), 2):
                    if pos[ia] >= 0 and pos[ib] >= 0 and \
                            abs(pos[ia] - pos[ib]) < motif_distance_cutoff:
                        clone_network_rows.append((a, b, "local", m["tag"]))
            else:
                for a, b in combinations(members, 2):
                    if all_aa_interchangeable:
                        clone_network_rows.append((a, b, "global", m["tag"]))
                    else:
                        tpos = m["tag"].find("%") + 1
                        if structboundaries:
                            tpos += boundary_size
                        if (a[tpos - 1:tpos] + b[tpos - 1:tpos]) in blosum:
                            clone_network_rows.append(
                                (a, b, "global", m["tag"]))
        # dedup
        clone_network_rows = list(dict.fromkeys(clone_network_rows))

    # singletons
    connected = set()
    for a, b, _, _ in clone_network_rows:
        connected.add(a)
        connected.add(b)
    singletons = [s for s in sequences["CDR3b"] if s not in connected]
    for i, s in enumerate(singletons, 1):
        clone_network_rows.append((s, s, "singleton", f"singleton_{i}"))
    connections = pd.DataFrame(
        clone_network_rows, columns=["V1", "V2", "type", "cluster_tag"])

    # ====================================================================
    # Part 4: Scoring
    # ====================================================================
    cluster_properties = None
    if merged_kept:
        props = pd.DataFrame([{
            "type": m["type"], "tag": m["tag"],
            "cluster_size": m["cluster_size"],
            "unique_cdr3_sample": m["unique_cdr3_sample"],
            "unique_cdr3_ref": m["unique_cdr3_ref"],
            "OvE": m["OvE"],
            "fisher.score": m["fisher.score"],
            "members": m["members"],
        } for m in merged_kept])
        props["fisher.score"] = [format_e1(x) for x in props["fisher.score"]]
        props["OvE"] = props["OvE"].replace([np.inf, -np.inf], 0)

        scoring = cluster_scoring(
            cluster_list=cluster_list, cdr3_sequences=cdr3_sequences,
            refdb_beta=refdb_beta, v_usage_freq=v_usage_freq,
            cdr3_length_freq=cdr3_length_freq,
            ref_cluster_size=ref_cluster_size, sim_depth=sim_depth,
            gliph_version=2, hla_cutoff=hla_cutoff,
            random_state=random_state)
        props = pd.concat([props.reset_index(drop=True),
                           scoring.reset_index(drop=True)], axis=1)
        order = [c for c in props.columns if c != "members"] + ["members"]
        cluster_properties = props[order]
        for c in cluster_properties.columns:
            if c in ("type", "tag", "members", "lowest.hlas"):
                continue
            try:
                cluster_properties[c] = pd.to_numeric(cluster_properties[c])
            except (ValueError, TypeError):
                pass

    parameters = {
        "gliph_version": 2, "ref_cluster_size": ref_cluster_size,
        "sim_depth": sim_depth, "lcminp": lcminp, "lcminove": lcminove,
        "motif_distance_cutoff": motif_distance_cutoff,
        "kmer_mindepth": kmer_mindepth,
        "accept_sequences_with_C_F_start_end":
            accept_sequences_with_C_F_start_end,
        "min_seq_length": min_seq_length, "structboundaries": structboundaries,
        "boundary_size": boundary_size, "motif_length": motif_length,
        "discontinuous_motifs": discontinuous_motifs,
        "local_similarities": local_similarities,
        "global_similarities": global_similarities,
        "global_vgene": global_vgene,
        "all_aa_interchangeable": all_aa_interchangeable,
        "cluster_min_size": cluster_min_size, "hla_cutoff": hla_cutoff,
    }

    return {
        "motif_enrichment": {"selected_motifs": part_1_res,
                             "all_motifs": part_1_all},
        "global_enrichment": part_2_res,
        "connections": connections,
        "cluster_properties": cluster_properties,
        "cluster_list": cluster_list,
        "parameters": parameters,
    }

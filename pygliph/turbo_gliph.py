"""GLIPH v1 -- ``turbo_gliph``.

Port of turboGliph's ``turbo_gliph`` (R/turbo_gliph_function_foreach.R),
following Glanville et al., Nature 2017. Local-motif significance is
assessed by repeated random sampling from a reference repertoire; global
similarity links CDR3s within a Hamming-distance cutoff.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from . import datasets
from .motifs import find_motifs
from .scoring import cluster_scoring

__all__ = ["turbo_gliph", "get_random_subsample"]


def get_random_subsample(
    refseqs_motif_region: Sequence[str],
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Draw an unbiased reference subsample of size ``n`` (no replacement).

    Port of ``getRandomSubsample`` for the default (unstratified) case.
    """
    idx = rng.choice(len(refseqs_motif_region), size=n, replace=False)
    return [refseqs_motif_region[i] for i in idx]


def _prepare_sequences(cdr3_sequences, global_vgene, vgene_stratify):
    if isinstance(cdr3_sequences, (list, tuple, pd.Series, np.ndarray)):
        seq = pd.DataFrame({"CDR3b": [str(x) for x in cdr3_sequences]})
        if global_vgene or vgene_stratify:
            raise ValueError("V-gene info required (column 'TRBV').")
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


def turbo_gliph(
    cdr3_sequences,
    refdb_beta="gliph_reference",
    ref_cluster_size: str = "original",
    v_usage_freq: pd.DataFrame | None = None,
    cdr3_length_freq: pd.DataFrame | None = None,
    sim_depth: int = 1000,
    lcminp: float = 0.01,
    lcminove=(1000, 100, 10),
    kmer_mindepth: int = 3,
    gccutoff: int | None = None,
    accept_sequences_with_C_F_start_end: bool = True,
    min_seq_length: int = 0,
    structboundaries: bool = True,
    boundary_size: int = 3,
    motif_length=(2, 3, 4),
    discontinuous: bool = False,
    local_similarities: bool = True,
    global_similarities: bool = True,
    global_vgene: bool = False,
    positional_motifs: bool = False,
    cdr3_len_stratify: bool = False,
    vgene_stratify: bool = False,
    public_tcrs: bool = True,
    cluster_min_size: int = 2,
    hla_cutoff: float = 0.1,
    random_state: int | None = 42,
) -> dict:
    """Identify TCR specificity groups with the original GLIPH algorithm.

    Parameters
    ----------
    cdr3_sequences
        CDR3b strings or a DataFrame with ``CDR3b`` and optional ``TRBV``,
        ``patient``, ``HLA``, ``counts``.
    sim_depth
        Number of reference resamplings for local-motif significance.
    lcminp, lcminove, kmer_mindepth
        Local-motif filters (max resampling p-value, min fold enrichment,
        min sample occurrences).
    gccutoff
        Maximum CDR3 Hamming distance for global similarity. ``None`` ->
        2 if fewer than 125 sequences else 1.
    positional_motifs
        Restrict local edges to identical motif position.
    public_tcrs
        If ``False``, clusters may not mix donors (needs ``patient``).
    random_state
        Seed for the resampling RNG.

    Returns
    -------
    dict
        Keys ``sample_log``, ``motif_enrichment``, ``connections``,
        ``cluster_properties``, ``cluster_list``, ``parameters``.
    """
    lcminove = list(lcminove)
    motif_length = list(motif_length)
    if structboundaries:
        min_seq_length = max(min_seq_length, 2 * boundary_size + 1)
    rng = np.random.default_rng(random_state)

    sequences = _prepare_sequences(cdr3_sequences, global_vgene,
                                   vgene_stratify)
    vgene_info = "TRBV" in sequences.columns
    patient_info = "patient" in sequences.columns
    if not patient_info and not public_tcrs:
        public_tcrs = True

    # ---- reference DB --------------------------------------------------
    if isinstance(refdb_beta, pd.DataFrame):
        ref = refdb_beta.copy().astype(str)
        if "CDR3b" not in ref.columns:
            ref = ref.rename(columns={ref.columns[0]: "CDR3b"})
        if "TRBV" not in ref.columns:
            ref["TRBV"] = ""
        ref = ref[["CDR3b", "TRBV"]]
    else:
        ref = datasets.load_reference_db(refdb_beta)[["CDR3b", "TRBV"]].copy()
    if accept_sequences_with_C_F_start_end:
        ref = ref[ref["CDR3b"].map(lambda s: bool(re.match(r"^C.*F$", s)))]
    ref = ref.drop_duplicates()
    ref = ref[ref["CDR3b"].str.len() >= min_seq_length]
    ref = ref[ref["CDR3b"].map(
        lambda s: bool(re.match(r"^[ACDEFGHIKLMNPQRSTVWY]*$", s)))]
    refseqs = ref["CDR3b"].to_list()

    # ---- sample motif region -------------------------------------------
    seqs_all = sequences["CDR3b"].to_list()
    if accept_sequences_with_C_F_start_end:
        seqs = [s for s in seqs_all if re.match(r"^C.*F$", s)]
    else:
        seqs = list(seqs_all)
    seqs = list(dict.fromkeys(seqs))
    seqs = [s for s in seqs if len(s) >= min_seq_length]
    if not seqs:
        raise ValueError("No CDR3b sequences passed the length/C..F filter.")
    if structboundaries:
        motif_region = [s[boundary_size:len(s) - boundary_size] for s in seqs]
        refseqs_motif_region = [s[boundary_size:len(s) - boundary_size]
                                for s in refseqs]
    else:
        motif_region = list(seqs)
        refseqs_motif_region = list(refseqs)

    # ====================================================================
    # Part 1 & 2: local-motif enrichment by resampling
    # ====================================================================
    sample_log = None
    selected_motifs = None
    all_motifs = None
    if local_similarities:
        discovery = find_motifs(motif_region, q=motif_length,
                                discontinuous=discontinuous)
        disc_counts = (discovery.groupby("motif", sort=False)["V1"].sum()
                       if len(discovery) else pd.Series(dtype=float))
        disc_motifs = list(disc_counts.index)
        actual = disc_counts.to_numpy(float)

        sim_matrix = np.zeros((sim_depth, len(disc_motifs)), dtype=float)
        for it in range(sim_depth):
            sub = get_random_subsample(refseqs_motif_region,
                                       len(motif_region), rng)
            sim = find_motifs(sub, q=motif_length, discontinuous=discontinuous)
            sim_counts = (sim.groupby("motif", sort=False)["V1"].sum()
                          if len(sim) else pd.Series(dtype=float))
            sim_matrix[it] = [sim_counts.get(m, 0.0) for m in disc_motifs]

        sample_log = pd.DataFrame(
            np.vstack([actual, sim_matrix]), columns=disc_motifs,
            index=["Discovery"] + [f"sim-{i}" for i in range(sim_depth)])

        # per-motif minimum fold change
        if len(lcminove) == 1:
            temp_minove = np.full(len(disc_motifs), lcminove[0], dtype=float)
        else:
            nam_len = np.asarray([len(m) for m in disc_motifs])
            has_dot = np.asarray(["." in m for m in disc_motifs])
            nam_len = nam_len - has_dot.astype(int)
            temp_minove = np.zeros(len(disc_motifs), dtype=float)
            for ml, ov in zip(motif_length, lcminove):
                temp_minove[nam_len == ml] = ov

        sel_rows, all_rows = [], []
        for i, motif in enumerate(disc_motifs):
            col = sim_matrix[:, i]
            pv = float(np.sum(col >= actual[i])) / sim_depth
            m = float(np.mean(col))
            mx = float(np.max(col))
            if m > 0:
                ove = actual[i] / m
            else:
                ove = 1.0 / (sim_depth * len(motif_region))
            rec = [motif, actual[i], round(m, 2), mx, round(ove, 3),
                   round(pv, 8)]
            all_rows.append(rec)
            if (ove >= temp_minove[i] and pv < lcminp
                    and actual[i] >= kmer_mindepth):
                pv2 = 1.0 / sim_depth if pv == 0 else pv
                sel_rows.append([motif, actual[i], round(m, 2), mx,
                                 round(ove, 3), round(pv2, 8)])
        cols = ["Motif", "counts", "avgRef", "topRef", "OvE", "p-value"]
        selected_motifs = pd.DataFrame(sel_rows, columns=cols)
        all_motifs = pd.DataFrame(all_rows, columns=cols)
        if len(selected_motifs) == 0:
            local_similarities = False

    # ====================================================================
    # Part 3: global similarity + clustering edges
    # ====================================================================
    if gccutoff is None:
        gccutoff = 2 if len(seqs) < 125 else 1

    temp_seqs, temp_vgenes = None, None
    if global_vgene:
        cf = [s for s in seqs_all if re.match(r"^C.*F$", s)] \
            if accept_sequences_with_C_F_start_end else list(seqs_all)
        temp_seqs = cf
        vmap = dict(zip(sequences["CDR3b"], sequences["TRBV"]))
        temp_vgenes = [vmap.get(s, "") for s in cf]

    clone_network = []
    not_in_global = set(range(len(seqs)))
    in_local = set()

    if global_similarities:
        # group by length for Hamming-distance comparisons
        by_len: dict[int, list[int]] = {}
        for i, mr in enumerate(motif_region):
            by_len.setdefault(len(mr), []).append(i)
        connected_global = set()
        for _, ids in by_len.items():
            for a, b in combinations(ids, 2):
                d = sum(1 for x, y in
                        zip(motif_region[a], motif_region[b]) if x != y)
                if d <= gccutoff:
                    if global_vgene:
                        va = {temp_vgenes[k] for k in range(len(temp_seqs))
                              if temp_seqs[k] == seqs[a]}
                        vb = {temp_vgenes[k] for k in range(len(temp_seqs))
                              if temp_seqs[k] == seqs[b]}
                        if not (va & vb):
                            continue
                    lo, hi = (a, b) if a < b else (b, a)
                    clone_network.append((seqs[lo], seqs[hi], "global"))
                    connected_global.add(a)
                    connected_global.add(b)
        not_in_global = set(range(len(seqs))) - connected_global

    if local_similarities and selected_motifs is not None:
        for motif in selected_motifs["Motif"]:
            local_ids = [i for i, mr in enumerate(motif_region)
                         if motif in mr]
            if len(local_ids) > 1:
                in_local.update(local_ids)
                for a, b in combinations(local_ids, 2):
                    if positional_motifs:
                        pa = motif_region[a].find(motif)
                        pb = motif_region[b].find(motif)
                        if pa != pb:
                            continue
                    clone_network.append((seqs[a], seqs[b], "local"))

    # public-TCR filter
    if not public_tcrs and patient_info:
        pat_map: dict[str, set] = {}
        for _, row in sequences.iterrows():
            pat_map.setdefault(row["CDR3b"], set()).add(row["patient"])
        clone_network = [(a, b, t) for (a, b, t) in clone_network
                         if pat_map.get(a, set()) & pat_map.get(b, set())]

    singleton_ids = sorted(not_in_global - in_local)
    for sid in singleton_ids:
        clone_network.append((seqs[sid], seqs[sid], "singleton"))

    connections = pd.DataFrame(clone_network, columns=["V1", "V2", "type"])

    # ====================================================================
    # Part 4: connected components
    # ====================================================================
    cluster_list: dict[str, pd.DataFrame] = {}
    cluster_properties = None
    if len(connections) > 0:
        import networkx as nx
        # expand to tuple-level edges (sequence + all metadata rows)
        seq_rows: dict[str, list[int]] = {}
        for ridx, cdr3 in enumerate(sequences["CDR3b"]):
            seq_rows.setdefault(cdr3, []).append(ridx)

        g = nx.Graph()
        sep = "$#$#$"
        ncol = sequences.shape[1]
        for a, b, t in clone_network:
            ids_a = seq_rows.get(a, [])
            ids_b = seq_rows.get(b, [])
            for ia in ids_a:
                ra = sequences.iloc[ia]
                key_a = sep.join(str(x) for x in ra)
                for ib in ids_b:
                    rb = sequences.iloc[ib]
                    if not public_tcrs and patient_info \
                            and ra["patient"] != rb["patient"]:
                        continue
                    if (global_vgene and vgene_info and t == "global"
                            and ra["TRBV"] != rb["TRBV"]):
                        continue
                    key_b = sep.join(str(x) for x in rb)
                    g.add_edge(key_a, key_b)

        comps = list(nx.connected_components(g))
        all_leaders: list[str] = []
        rows = []
        for comp in comps:
            parsed = [node.split(sep) for node in comp]
            mdf = pd.DataFrame(parsed, columns=sequences.columns)
            mdf = mdf.drop_duplicates().reset_index(drop=True)
            members = sorted(set(mdf["CDR3b"]))
            csize = len(members)
            leader = f"CRG-{members[0]}"
            t = leader
            c = 1
            while t in all_leaders:
                t = f"{leader}-{c}"
                c += 1
            leader = t
            all_leaders.append(leader)
            rows.append({"cluster_size": csize, "tag": leader,
                         "members": " ".join(members)})
            cluster_list[leader] = mdf

        cluster_properties = pd.DataFrame(rows)
        # drop small clusters
        keep = cluster_properties["cluster_size"] >= cluster_min_size
        dropped = cluster_properties[~keep]["tag"].to_list()
        for tag in dropped:
            cluster_list.pop(tag, None)
        cluster_properties = cluster_properties[keep].reset_index(drop=True)
        if len(cluster_properties) == 0:
            cluster_properties = None
            cluster_list = {}

    # ====================================================================
    # Part 5: scoring
    # ====================================================================
    if cluster_list:
        scoring = cluster_scoring(
            cluster_list=cluster_list, cdr3_sequences=cdr3_sequences,
            refdb_beta=refdb_beta, v_usage_freq=v_usage_freq,
            cdr3_length_freq=cdr3_length_freq,
            ref_cluster_size=ref_cluster_size, sim_depth=sim_depth,
            gliph_version=1, hla_cutoff=hla_cutoff,
            random_state=random_state)
        cluster_properties = pd.concat(
            [cluster_properties.reset_index(drop=True),
             scoring.reset_index(drop=True)], axis=1)
        order = [c for c in cluster_properties.columns
                 if c != "members"] + ["members"]
        cluster_properties = cluster_properties[order]

    parameters = {
        "gliph_version": 1, "ref_cluster_size": ref_cluster_size,
        "sim_depth": sim_depth, "lcminp": lcminp, "lcminove": lcminove,
        "kmer_mindepth": kmer_mindepth,
        "accept_sequences_with_C_F_start_end":
            accept_sequences_with_C_F_start_end,
        "min_seq_length": min_seq_length, "gccutoff": gccutoff,
        "structboundaries": structboundaries, "boundary_size": boundary_size,
        "motif_length": motif_length, "discontinuous": discontinuous,
        "local_similarities": local_similarities,
        "global_similarities": global_similarities,
        "global_vgene": global_vgene, "positional_motifs": positional_motifs,
        "cdr3_len_stratify": cdr3_len_stratify,
        "vgene_stratify": vgene_stratify, "public_tcrs": public_tcrs,
        "cluster_min_size": cluster_min_size, "hla_cutoff": hla_cutoff,
    }

    return {
        "sample_log": sample_log,
        "motif_enrichment": {"selected_motifs": selected_motifs,
                             "all_motifs": all_motifs},
        "connections": connections,
        "cluster_properties": cluster_properties,
        "cluster_list": cluster_list,
        "parameters": parameters,
    }

"""Reading and writing GLIPH / GLIPH2 result folders.

Mirrors turboGliph's ``load_gliph_output`` and the file layout written by
``turbo_gliph`` / ``gliph2`` when a ``result_folder`` is supplied.
"""
from __future__ import annotations

import os

import pandas as pd

__all__ = ["save_gliph_output", "load_gliph_output"]


def _write(df, path):
    if df is None:
        return
    df.to_csv(path, sep="\t", index=False)


def save_gliph_output(result: dict, result_folder: str) -> None:
    """Write a GLIPH / GLIPH2 result dict to a folder of TSV files.

    Parameters
    ----------
    result
        Output of :func:`pygliph.gliph2` or :func:`pygliph.turbo_gliph`.
    result_folder
        Destination directory (created if absent).
    """
    os.makedirs(result_folder, exist_ok=True)
    params = result.get("parameters", {})
    version = params.get("gliph_version", 2)

    me = result.get("motif_enrichment") or {}
    _write(me.get("selected_motifs"),
           os.path.join(result_folder, "local_similarities.txt"))
    _write(me.get("all_motifs"),
           os.path.join(result_folder, "all_motifs.txt"))
    if version == 1:
        _write(result.get("sample_log"),
               os.path.join(result_folder, "kmer_resample_log.txt"))
    else:
        _write(result.get("global_enrichment"),
               os.path.join(result_folder, "global_similarities.txt"))

    conn = result.get("connections")
    if conn is not None:
        conn.to_csv(os.path.join(result_folder, "clone_network.txt"),
                    sep="\t", index=False, header=False)
    _write(result.get("cluster_properties"),
           os.path.join(result_folder, "convergence_groups.txt"))

    cl = result.get("cluster_list") or {}
    if cl:
        frames = []
        for tag, df in cl.items():
            tmp = df.copy()
            tmp.insert(0, "tag", tag)
            frames.append(tmp)
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(result_folder, "cluster_member_details.txt"),
            sep="\t", index=False)

    # parameters
    with open(os.path.join(result_folder, "parameter.txt"), "w") as fh:
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            fh.write(f"{k}\t{v}\n")


def load_gliph_output(result_folder: str) -> dict:
    """Load a GLIPH / GLIPH2 result folder back into a dict.

    Parameters
    ----------
    result_folder
        Folder previously written by :func:`save_gliph_output`.

    Returns
    -------
    dict
        Same structure as :func:`pygliph.gliph2` / :func:`pygliph.turbo_gliph`.
    """
    if not os.path.isdir(result_folder):
        raise FileNotFoundError(f"No such folder: {result_folder}")

    def _read(name, **kw):
        p = os.path.join(result_folder, name)
        if os.path.exists(p):
            return pd.read_csv(p, sep="\t", **kw)
        return None

    parameters: dict = {}
    pp = os.path.join(result_folder, "parameter.txt")
    if os.path.exists(pp):
        with open(pp) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                key, val = parts[0], parts[1]
                try:
                    parameters[key] = float(val) if "." in val else int(val)
                except ValueError:
                    parameters[key] = val

    selected = _read("local_similarities.txt")
    all_motifs = _read("all_motifs.txt")
    cluster_props = _read("convergence_groups.txt")
    global_enr = _read("global_similarities.txt")

    cl_df = _read("cluster_member_details.txt")
    cluster_list: dict = {}
    if cl_df is not None and "tag" in cl_df.columns:
        for tag, sub in cl_df.groupby("tag"):
            cluster_list[str(tag)] = sub.drop(columns=["tag"]).reset_index(
                drop=True)

    conn_path = os.path.join(result_folder, "clone_network.txt")
    connections = None
    if os.path.exists(conn_path):
        connections = pd.read_csv(conn_path, sep="\t", header=None)

    version = parameters.get("gliph_version", 2)
    if version == 1:
        return {
            "sample_log": _read("kmer_resample_log.txt", index_col=0),
            "motif_enrichment": {"selected_motifs": selected,
                                 "all_motifs": all_motifs},
            "connections": connections,
            "cluster_properties": cluster_props,
            "cluster_list": cluster_list,
            "parameters": parameters,
        }
    return {
        "motif_enrichment": {"selected_motifs": selected,
                             "all_motifs": all_motifs},
        "global_enrichment": global_enr,
        "connections": connections,
        "cluster_properties": cluster_props,
        "cluster_list": cluster_list,
        "parameters": parameters,
    }

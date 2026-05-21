"""Shared fixtures and the R-reference build hook for pygliph tests."""
from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
R_OUT = os.path.join(HERE, "R_out")
R_DRIVER = os.path.join(HERE, "r_reference_driver.R")

# CMAP path-based R environment (override with PYGLIPH_RSCRIPT).
_RSCRIPT = os.environ.get(
    "PYGLIPH_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript")


def _r_outputs_present() -> bool:
    needed = ["input.tsv", "find_motifs_cont.tsv", "gliph2_all_motifs.tsv",
              "gliph2_selected.tsv", "gliph2_clusters.tsv"]
    return all(os.path.exists(os.path.join(R_OUT, f)) for f in needed)


def _build_r_reference() -> bool:
    """Run the turboGliph R driver if R is available; return success."""
    if not shutil.which(_RSCRIPT) and not os.path.exists(_RSCRIPT):
        return False
    env = dict(os.environ)
    # the CMAP env needs a modern gcc on PATH for some packages
    gcc = "/share/software/user/open/gcc/14.2.0/bin"
    if os.path.isdir(gcc):
        env["PATH"] = gcc + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = (
            "/share/software/user/open/gcc/14.2.0/lib64" + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""))
    try:
        subprocess.run([_RSCRIPT, R_DRIVER, R_OUT], check=True, env=env,
                       capture_output=True, timeout=900)
    except Exception:
        return False
    return _r_outputs_present()


@pytest.fixture(scope="session")
def r_reference():
    """Path to the directory of turboGliph reference outputs.

    Skips the test if the R reference cannot be produced.
    """
    if not _r_outputs_present():
        if not _build_r_reference():
            pytest.skip("turboGliph R reference unavailable")
    return R_OUT


@pytest.fixture(scope="session")
def r_input(r_reference):
    """The exact input subset shared by the R and Python sides."""
    return pd.read_csv(os.path.join(r_reference, "input.tsv"), sep="\t",
                       dtype=str, keep_default_na=False)

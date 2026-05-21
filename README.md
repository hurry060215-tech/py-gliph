# py-gliph

Pure-Python port of the R package
**[turboGliph](https://github.com/HetzDra/turboGliph)** — implementations
of **GLIPH** (Glanville *et al.*, *Nature* 2017) and **GLIPH2** (Huang
*et al.*, *Nat. Biotechnol.* 2020): **G**rouping of **L**ymphocyte
**I**nteractions by **P**aratope **H**otspots.

`pygliph` clusters T-cell receptors (TCRs) into **specificity groups**
predicted to bind the same MHC-restricted peptide antigen, based on
shared CDR3 motifs. It is a standalone, dependency-light re-implementation
that does **not** require R or `rpy2`.

| | |
|---|---|
| PyPI / import name | `pygliph` |
| Repository | `omicverse/py-gliph` |
| License | Apache-2.0 |
| Upstream | turboGliph 0.99.2 (GPL-3, R) |
| Numerical parity | deterministic parts bit-exact vs turboGliph |

## Install

```bash
pip install pygliph              # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `networkx` (and `matplotlib`
for `plot_network`, an optional extra).

## What it does

GLIPH/GLIPH2 build specificity groups from two kinds of CDR3 similarity:

* **Local (motif) similarity** — enriched short continuous CDR3 motifs
  (2–4-mers). GLIPH2 scores enrichment with Fisher's exact test
  (hypergeometric) against a naive reference TCR pool; motif clusters
  are position-restricted (a motif may shift by < 3 residues).
  GLIPH v1 instead uses repeated random sampling from the reference.
* **Global similarity** — CDR3s of equal length differing by a single
  amino acid (Hamming distance 1). GLIPH2 uses position-specific
  structures with an optional BLOSUM62 interchangeability constraint.
* **Specificity-group construction** — connected components of the
  combined local + global similarity graph.
* **Per-group scoring** — network size, CDR3-length restriction,
  V-gene bias, clonal-expansion enrichment and shared-HLA enrichment
  combined into a `total.score` (GLIPH2's `convergence_groups.txt`).
* **N/P-residue up-weighting** — local cluster significance is boosted
  for motifs overlapping non-germline (N/P-nucleotide) encoded residues.

## Quick start

```python
import pygliph as pg

# bundled example: ~2000 TCRs of known specificity
data = pg.datasets.load_gliph_input_data()

# GLIPH2
res = pg.gliph2(data, sim_depth=1000)
res["cluster_properties"]            # scored specificity groups
res["motif_enrichment"]["selected_motifs"]   # enriched local motifs
res["global_enrichment"]             # global-similarity structures
res["cluster_list"]                  # {tag: member DataFrame}

# original GLIPH (v1)
res1 = pg.turbo_gliph(data, sim_depth=1000)

# configurable hybrid
res2 = pg.gliph_combined(data, local_method="fisher", global_method="fisher")
```

## API

| Function | Purpose |
|---|---|
| `gliph2` | GLIPH2 algorithm — Fisher-test local motifs + global structures |
| `turbo_gliph` | original GLIPH — resampling local motifs + Hamming globals |
| `gliph_combined` | configurable hybrid of GLIPH / GLIPH2 |
| `cluster_scoring` | per-cluster enrichment scoring |
| `find_motifs` | continuous / discontinuous motif enumeration |
| `de_novo_TCRs` | de-novo CDR3 generation from a specificity group |
| `plot_network` | specificity-group network visualisation |
| `save_gliph_output` / `load_gliph_output` | result-folder I/O |
| `datasets` | bundled example data and reference resources |

Each result is a dict mirroring turboGliph: `motif_enrichment`,
`global_enrichment`, `connections`, `cluster_properties`,
`cluster_list`, `parameters` (and `sample_log` for GLIPH v1).

## Input format

A `pandas.DataFrame` (or a plain list of CDR3b strings) with column
`CDR3b` and any of the optional columns `TRBV`, `patient`, `HLA`,
`counts` — the same schema as turboGliph.

## R parity

`tests/` runs the **same input** through turboGliph (R) and `pygliph`
and asserts agreement. The deterministic parts are **bit-exact**:

* `find_motifs` continuous and discontinuous motif counts;
* GLIPH2 local-motif hypergeometric ("Fisher") p-values, fold change,
  selected motifs and their position ranges;
* GLIPH2 global-similarity structures and their Fisher scores;
* the clone-network edge list (directed 4-tuples);
* cluster membership, sizes and cluster-level Fisher scores;
* `de_novo_TCRs` position weight matrices and sample-sequence scores.

**Unavoidable approximate parts:** any score derived from random
resampling — GLIPH v1's repeated-random-sampling local-motif selection,
and the `total.score`/length/V-gene/clonal-expansion sub-scores in both
algorithms — depends on the RNG. R's `sample.int` (Mersenne-Twister)
and NumPy's PCG64 produce different draws, so these are checked for
high set / rank agreement (correlation > 0.95) rather than bit equality.
Fix `random_state` for reproducible Python runs.

## References

* Glanville, J. *et al.* Identifying specificity groups in the T cell
  receptor repertoire. *Nature* **547**, 94–98 (2017).
* Huang, H. *et al.* Analyzing the *Mycobacterium tuberculosis* immune
  response by T-cell receptor clustering with GLIPH2 and genome-wide
  antigen screening. *Nat. Biotechnol.* **38**, 1194–1202 (2020).
* turboGliph: <https://github.com/HetzDra/turboGliph>

## License

Apache-2.0. The upstream R package turboGliph is GPL-3; `pygliph` is an
independent re-implementation from the published algorithms and the
turboGliph source, and ships the small reference data tables exported
from turboGliph for parity testing.

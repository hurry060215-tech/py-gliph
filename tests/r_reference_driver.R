#!/usr/bin/env Rscript
# Drive the R package turboGliph on the bundled gliph_input_data so the
# Python port (pygliph) can be checked against it.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# Outputs (TSV files in out_dir):
#   input.tsv                 the exact input subset used by both sides
#   find_motifs_cont.tsv      find_motifs() continuous 2/3/4-mers
#   find_motifs_disc.tsv      find_motifs() with discontinuous motifs
#   gliph2_all_motifs.tsv     gliph2() local-motif Fisher table (all motifs)
#   gliph2_selected.tsv       gliph2() significantly enriched motifs
#   gliph2_global.tsv         gliph2() global-similarity structures
#   gliph2_connections.tsv    gliph2() clone network edge list
#   gliph2_clusters.tsv       gliph2() convergence_groups (with scores)
#   gliph1_selected.tsv       turbo_gliph() selected motifs
#   gliph1_clusters.tsv       turbo_gliph() convergence groups

suppressPackageStartupMessages(library(turboGliph))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

utils::data("gliph_input_data", package = "turboGliph")

# Use a deterministic subset; drop the uninformative TRBV/HLA placeholders
# so deterministic parts (motifs / Fisher / globals) are isolated.
N <- 300
sample_in <- gliph_input_data[seq_len(N), c("CDR3b", "patient", "counts")]
write.table(sample_in, file.path(out_dir, "input.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

seqs <- as.character(sample_in$CDR3b)

# --- find_motifs ----------------------------------------------------
fm <- find_motifs(seqs = seqs, q = 2:4)
colnames(fm) <- c("motif", "count")
fm <- fm[order(fm$motif), ]
write.table(fm, file.path(out_dir, "find_motifs_cont.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

fmd <- find_motifs(seqs = seqs, q = 2:3, discontinuous = TRUE)
colnames(fmd) <- c("motif", "count")
fmd <- fmd[order(fmd$motif), ]
write.table(fmd, file.path(out_dir, "find_motifs_disc.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- gliph2 (Fisher local + global) ---------------------------------
set.seed(42)
g2 <- gliph2(cdr3_sequences = sample_in,
             sim_depth = 100,
             n_cores = 1)

write.table(g2$motif_enrichment$all_motifs,
            file.path(out_dir, "gliph2_all_motifs.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(g2$motif_enrichment$selected_motifs,
            file.path(out_dir, "gliph2_selected.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
if (!is.null(g2$global_enrichment)) {
  write.table(g2$global_enrichment,
              file.path(out_dir, "gliph2_global.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}
write.table(g2$connections, file.path(out_dir, "gliph2_connections.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(g2$cluster_properties,
            file.path(out_dir, "gliph2_clusters.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- turbo_gliph (GLIPH v1) -----------------------------------------
set.seed(42)
g1 <- turbo_gliph(cdr3_sequences = sample_in,
                  sim_depth = 100,
                  n_cores = 1)
write.table(g1$motif_enrichment$selected_motifs,
            file.path(out_dir, "gliph1_selected.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(g1$cluster_properties,
            file.path(out_dir, "gliph1_clusters.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- de_novo_TCRs (deterministic PWM) -------------------------------
for (tg in names(g2$cluster_list)) {
  cl <- g2$cluster_list[[tg]]
  if (sum(nchar(cl$CDR3b) >= 10) >= 3) {
    set.seed(7)
    dn <- de_novo_TCRs(convergence_group_tag = tg, clustering_output = g2,
                       sims = 2000, num_tops = 20, n_cores = 1)
    write.table(dn$PWM_Scoring, file.path(out_dir, "denovo_pwm.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(dn$sample_sequences_scores,
                file.path(out_dir, "denovo_sample_scores.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
    writeLines(tg, file.path(out_dir, "denovo_tag.txt"))
    break
  }
}

cat("turboGliph R reference done\n")

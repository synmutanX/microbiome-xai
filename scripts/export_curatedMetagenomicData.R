#!/usr/bin/env Rscript
# Export every public colorectal-cancer (CRC) stool metagenome in
# curatedMetagenomicData (Pasolli et al. 2017, Nat Methods; MetaPhlAn profiles)
# to two TSV files that mbxai can read.
#
# Setup (once):
#   install.packages("BiocManager"); BiocManager::install(c("curatedMetagenomicData", "dplyr", "tidyr"))
# Run:
#   Rscript scripts/export_curatedMetagenomicData.R data/cmd_crc
# Then:
#   mbxai run configs/crc_curatedMetagenomicData.yaml --out results/crc
#
# Output:
#   relative_abundance.tsv   taxa (full lineage, '|' separated) x samples
#   metadata.tsv             one row per sample (study_name, study_condition, age, BMI, country, ...)
#   studies.tsv              cases / controls per study, to choose cohorts for the config

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(dplyr)
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "data/cmd_crc"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

crc_studies <- sampleMetadata |>
  filter(study_condition == "CRC") |>
  distinct(study_name) |>
  pull(study_name)

meta <- sampleMetadata |>
  filter(study_name %in% crc_studies,
         body_site == "stool",
         study_condition %in% c("CRC", "control")) |>
  group_by(study_name, subject_id) |>   # one sample per person
  slice(1) |>
  ungroup() |>
  select(where(~ !all(is.na(.x))))

se <- returnSamples(meta, dataType = "relative_abundance", counts = FALSE, rownames = "long")

abundance <- as.data.frame(assay(se))
write.table(cbind(taxon = rownames(abundance), abundance), file.path(out, "relative_abundance.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

md <- as.data.frame(colData(se))
md[] <- lapply(md, function(x) if (is.list(x)) vapply(x, paste, character(1), collapse = ";") else x)
write.table(cbind(sample_id = rownames(md), md), file.path(out, "metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

studies <- md |>
  count(study_name, country, study_condition) |>
  tidyr::pivot_wider(names_from = study_condition, values_from = n, values_fill = 0)
write.table(studies, file.path(out, "studies.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
print(studies)
message("Exported ", ncol(abundance), " samples to ", out)

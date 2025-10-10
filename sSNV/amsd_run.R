# To run AMSD on SMaHT sample COLO829TB

library(tidyverse)
library(mutspecdist)
library(sigfit)
library(ggrepel)
# setwd("C:/Users/sfhar/Dropbox (Personal)/Personal Files/Harris-Feder/AMSD/smaht")

##########################
# Load files

files <- list.files("colo_spectrum_09132025", full.names = TRUE)

# --- COLO tables ---
colo_files <- files[grepl("COLO829TB", files) & !grepl("DBS78", files)]
colo_tables <- colo_files %>%
  set_names(basename(.)) %>%
  map(~ read_tsv(.x, show_col_types = FALSE))

# Rename each value column to the filename
colo_tables <- imap(colo_tables, ~ rename(.x, !!.y := 2))

# Merge all by MutationType
COLO829TB_merged <- reduce(colo_tables, full_join, by = "MutationType")


# --- Kmer tables ---
kmer_files <- files[grepl("kmer", files)]
kmer_tables <- kmer_files %>%
  set_names(basename(.)) %>%
  map(~ read_tsv(.x, show_col_types = FALSE))

# Merge by "3mer"
kmer_merged <- reduce(kmer_tables, full_join, by = "3mer")

# Put them in order
COLO829TB_ordered <- COLO829TB_merged %>%
  separate(MutationType, into = c("a","b"), sep = "\\[", remove = FALSE) %>%
  separate(b, into = c("b","c"), sep = "\\]") %>%
  arrange(b,a,c) %>%
  select(-a,-b,-c) %>%
  column_to_rownames(var = "MutationType") %>%
  t() %>%
  as.data.frame()

##########################
# Calculate kmer mutational opportunities and correct sample spectra

sample_counts <- COLO829TB_ordered

kmer_C <- kmer_merged %>%
  filter(str_sub(`3mer`, 2, 2) == "C")

kmer_T <- kmer_merged %>%
  filter(str_sub(`3mer`, 2, 2) == "T")

oppertunities <- rbind(kmer_C,kmer_C,kmer_C,kmer_T,kmer_T,kmer_T) %>%
  select(-`3mer`) %>%
  t() %>%
  as.data.frame()/3 # correction for tripling everything
colnames(oppertunities) <- colnames(sample_counts )

samples_ordered <- COLO829TB_ordered %>%
  rownames_to_column(var = "name") %>%
  separate(name, into = c("name", NA, NA), sep = "\\.") %>%
  separate(name, into = c(NA,"type"), sep = "Pruned_") %>%
  filter(!is.na(type)) %>%
  column_to_rownames(var = "type")

######################
#  Correct for mutational oppertunities
  non_sat_list <- intersect(rownames(oppertunities), rownames(samples_ordered))
  sat_list <- intersect(paste0("Satellite_",rownames(oppertunities)), rownames(samples_ordered)) %>%
    sub("^Satellite_", "", .)
  
# Non-satellite
  oppertunities_ordered <- oppertunities[non_sat_list,] 
  samples_ordered2 <- samples_ordered[non_sat_list,] 
  full_genome_oppertunities <- oppertunities[rep("count", nrow(oppertunities_ordered)),] # full genome kmers
  corrected <- samples_ordered2*full_genome_oppertunities/oppertunities_ordered # correct each sample for full genome kmers
  corrected_spectra <- corrected/rowSums(corrected)*rowSums(samples_ordered2) # correct to make sure mutaion counts add up same
  corrected_spectra_rounded <- round(corrected_spectra) # round to integer

  plot_spectrum(oppertunities_ordered, pdf_path = "oppertunities2.pdf")
  plot_spectrum(corrected_spectra_rounded, pdf_path = "corrected_spectra_rounded2.pdf")
  plot_spectrum(samples_ordered2, pdf_path = "spectra2.pdf")

# Satellite
  oppertunities_satellite <- oppertunities[sat_list,] 
  samples_satellite <- samples_ordered[paste0("Satellite_",sat_list),] 
  full_genome_oppertunities_satellite <- oppertunities[rep("count", nrow(oppertunities_satellite)),] # full genome kmers
  corrected_satellite <- samples_satellite*full_genome_oppertunities_satellite/oppertunities_satellite # correct each sample for full genome kmers
  corrected_spectra_satellite <- corrected_satellite/rowSums(corrected_satellite)*rowSums(samples_satellite) # correct to make sure mutaion counts add up same
  corrected_spectra_rounded_satellite <- round(corrected_spectra_satellite) # round to integer  
  
  plot_spectrum(oppertunities_satellite, pdf_path = "oppertunities_satellite.pdf")
  plot_spectrum(corrected_spectra_rounded_satellite, pdf_path = "corrected_spectra_rounded_satellite2.pdf")
  plot_spectrum(samples_satellite, pdf_path = "spectra_satellite.pdf")  

# CDR
  oppertunities_CDR <- oppertunities[c("CDR","Non-CDR"),] 
  samples_CDR <- samples_ordered[c("CDR_BL_50kb","NON_CDR_BL_50kb"),] 
  full_genome_oppertunities_CDR <- oppertunities[rep("count", nrow(oppertunities_CDR)),] # full genome kmers
  corrected_CDR <- samples_CDR*full_genome_oppertunities_CDR/oppertunities_CDR # correct each sample for full genome kmers
  corrected_spectra_CDR <- corrected_CDR/rowSums(corrected_CDR)*rowSums(samples_CDR) # correct to make sure mutaion counts add up same
  corrected_spectra_rounded_CDR <- round(corrected_spectra_CDR) # round to integer  
  
  plot_spectrum(oppertunities_CDR, pdf_path = "oppertunities_CDR.pdf")
  plot_spectrum(corrected_spectra_rounded_CDR, pdf_path = "corrected_spectra_rounded_CDR.pdf")
  plot_spectrum(samples_CDR, pdf_path = "spectra_CDR.pdf")    
  
##########################
# Subset into individual mutations before and after correcting for mutational opportunities

# function to participation into multiple samples
partition_mutations_fast <- function(mat, group_size = 10, min_total = 100) {
  # mat: data.frame or matrix with rows = samples, cols = mutation types (SBS96)
  # group_size: number of mutations per partition
  # min_total: only process rows with >= this many mutations
  
  results <- vector("list", nrow(mat))
  sample_names <- rownames(mat)
  mut_types <- colnames(mat)
  
  for (s in seq_len(nrow(mat))) {
    row_counts <- mat[s, ]
    total_mut <- sum(row_counts)
    
    if (total_mut < min_total) {
      next
    }
    
    # Number of groups
    n_groups <- ceiling(total_mut / group_size)
    
    # Instead of expanding, assign group IDs directly
    group_ids <- sample(rep(seq_len(n_groups), each = group_size, length.out = total_mut))
    
    # Create matrix to hold group counts
    group_counts <- matrix(0, nrow = n_groups, ncol = length(mut_types),
                           dimnames = list(NULL, mut_types))
    
    # Efficiently distribute counts to groups
    mut_index <- rep(seq_along(mut_types), times = row_counts)
    for (i in seq_along(mut_index)) {
      group_counts[group_ids[i], mut_index[i]] <- group_counts[group_ids[i], mut_index[i]] + 1
    }
    
    # Convert to tibble
    df <- as.data.frame(group_counts)
    df$Sample <- sample_names[s]
    df$Group <- seq_len(n_groups)
    df$n_mutations <- rowSums(group_counts)
    
    results[[s]] <- tibble::as_tibble(df)
  }
  
  dplyr::bind_rows(results)
}



# Run on corrected
  partitioned_corrected <- partition_mutations_fast(corrected_spectra_rounded, group_size = 1, min_total = 100)
  partitioned_corrected_sat <- partition_mutations_fast(corrected_spectra_rounded_satellite, group_size = 1, min_total = 10)
  partitioned_uncorrected_CDR <- partition_mutations_fast(samples_CDR, group_size = 1, min_total = 10)
  partitioned_corrected_CDR <- partition_mutations_fast(corrected_spectra_rounded_CDR, group_size = 1, min_total = 10)

# Check results
  partitioned_corrected
  partitioned_corrected_sat
  partitioned_corrected_CDR 
##########################
# run AMSD
run_amsd_loop <- function(df, group_col, ref = "all", sims = 10000) {
  group_col <- ensym(group_col)
  
  # Pre-filter for mutation count, but KEEP grouping column
  df_filtered <- df %>%
    #filter(n_mutations > 1) %>%
    select(-Group, -n_mutations)
  
  # Build reference set ONCE
  if (ref == "all") {
    set2 <- df_filtered %>% select(-!!group_col)
  } else {
    set2 <- df_filtered %>%
      filter(!!group_col == ref) %>%
      select(-!!group_col)
  }
  
  # Iterate
  results <- map_dfr(unique(dplyr::pull(df, !!group_col)), function(repeat1) {
    message("starting ", repeat1)
    
    set1 <- df_filtered %>%
      filter(!!group_col == repeat1) %>%
      select(-!!group_col)
    
    # skip if no rows left
    if (nrow(set1) == 0 || nrow(set2) == 0) {
      return(NULL)
    }
    
    amsd_output <- amsd(set1, set2, mean_or_sum = "mean",
                        seed = 1234, n_sim = sims)
    
    tibble(
      pvalue = amsd_output$p,
      cosine_dist = amsd_output$cosine,
      group = repeat1
    )
  })
  
  return(results)
}

sims= 100000


# Case 1: corrected, each type vs genome-wide mutations
results_corr_GW <- run_amsd_loop(partitioned_corrected, Sample, ref = "all", sims = sims)
print("done with vs genome-wide mutations")
results_corr_GW

# Case 2: corrected, each Sample vs non-repeat
results_corr_NR <- run_amsd_loop(partitioned_corrected, Sample, ref = "None_RE", sims = sims)
print("done with vs non-repeat")
results_corr_NR

sims= 10000

# Case 3: corrected, each type vs genome-wide mutations
results_corr_sat <- run_amsd_loop(partitioned_corrected_sat, Sample, ref = "all", sims = sims)
print("done with sat")
#results_corr_sat$n_muts <- rowSums(corrected_spectra_satellite)
results_corr_sat

# Case 4: CDR comparison
results_uncorr_CDR <- run_amsd_loop(partitioned_uncorrected_CDR, Sample, ref = "CDR_BL_50kb", sims = sims)
results_corr_CDR <- run_amsd_loop(partitioned_corrected_CDR, Sample, ref = "CDR_BL_50kb", sims = sims)
results_uncorr_CDR
results_corr_CDR

################################
# plot and output result

merged <- bind_rows(
  results_corr_GW %>% 
    mutate(correction = "corrected", comparison = "vs genome-wide mutations"),
  results_corr_NR %>% 
    mutate(correction = "corrected", comparison = "vs non-repeat"),
  results_corr_sat %>% 
    mutate(correction = "corrected", comparison = "vs all Satellite_"),
  results_uncorr_CDR %>% 
    mutate(correction = "uncorrected", comparison = "vs CDR_BL_50kb"),
  results_corr_CDR %>% 
    mutate(correction = "corrected", comparison = "vs CDR_BL_50kb")
)

# Save as TSV
write_tsv(merged, "merged_amsd_results_09132025.tsv")

# plot_amsd_results <- function(results, sims, title, exclude_group = NULL, ref_results = NULL) {
#   
#   # Optionally filter out a group
#   if (!is.null(exclude_group)) {
#     results <- results %>% filter(group != exclude_group)
#   }
#   
#   # number of tests (Bonferroni correction)
#   n_tests <- nrow(results)
#   if (!is.null(ref_results)) {
#     # allow using another results object to set n_tests if desired
#     n_tests <- nrow(ref_results)
#   }
#   
#   # thresholds
#   sim_thresh   <- -log10(1 / sims)
#   bonf_thresh  <- -log10(0.05 / n_tests)
#   p05_thresh   <- -log10(0.05)
#   
#   # make plot
#   ggplot(results, aes(cosine_dist, -log10(pvalue), label = group)) +
#     geom_point() +
#     geom_hline(yintercept = sim_thresh) +
#     geom_hline(yintercept = bonf_thresh, linetype = "dashed") +
#     geom_hline(yintercept = p05_thresh, linetype = "dotted") +
#     # labels
#     annotate("text", x = max(results$cosine_dist, na.rm=TRUE), y = sim_thresh, 
#              label = "p = 1/permutaions", vjust = -0.5, hjust = 1, size = 3.5) +
#     annotate("text", x = max(results$cosine_dist, na.rm=TRUE), y = bonf_thresh, 
#              label = "Bonf. 0.05/n", vjust = -0.5, hjust = 1, size = 3.5) +
#     annotate("text", x = max(results$cosine_dist, na.rm=TRUE), y = p05_thresh, 
#              label = "p = 0.05", vjust = -0.5, hjust = 1, size = 3.5) +
#     ylim(0, max(4.5, sim_thresh, bonf_thresh, p05_thresh)) +
#     geom_label_repel() +
#     theme_classic() +
#     ggtitle(title)
# }
# 
# plot_amsd_results(results_corr_sat)
# 
# # plot_amsd_results(results_uncorr_GW, sims, 
# #                   title = "Result from AMSD, NOT corrected for kmers\nvs genome-wide mutation spectra") %>%
# #   ggsave("results_uncorr_GW.pdf", plot = ., width = 6, height = 4)
# plot_amsd_results(results_corr_GW, sims, 
#                   title = "Result from AMSD, NOT corrected for kmers\nvs genome-wide mutation spectra") %>%
#   ggsave("results_corr_GW_09132025.pdf", plot = ., width = 6, height = 4)
# # plot_amsd_results(results_uncorr_NR, sims, 
# #                   title = "Result from AMSD, corrected for kmer content\nvs non-repeat mutation spectra", 
# #                   exclude_group = "None_RE") %>%
# #   ggsave("results_uncorr_NR.pdf", plot = ., width = 6, height = 4)
# plot_amsd_results(results_corr_NR, sims, 
#                   title = "Result from AMSD, corrected for kmer content\nvs non-repeat mutation spectra", 
#                   exclude_group = "None_RE") %>%
#   ggsave("results_corr_NR_09132025.pdf", plot = ., width = 6, height = 4)

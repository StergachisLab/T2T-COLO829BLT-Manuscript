# %%
%%HTML
<style>
    body {
        --vscode-font-family: "CaskaydiaCove Nerd Font"
    }
</style>

# %%
import gzip as gz
import io
import os
import random
import string
from glob import glob
from itertools import chain
from webbrowser import get
import polars as pl
import pyranges as pr
import matplotlib.pyplot as plt
import matplotlib.ticker
from plotnine import *
import numpy as np
import pandas as pd
import seaborn as sns
from mizani.formatters import scientific_format, comma_format
from scipy.stats import fisher_exact
from scipy.stats import poisson
from scipy.stats import chi2_contingency
from scipy.stats import linregress
from scipy.spatial import distance
import statsmodels.api as sm
from statsmodels.stats import multitest
from statsmodels.stats.proportion import proportions_ztest

sns.set_theme(font="Arial", font_scale=1.15, style="ticks")
matplotlib.rcParams["figure.dpi"] = 300
plt.rc("axes.spines", top=False, right=False)
pl.Config.set_tbl_width_chars(10000)
pl.Config.set_fmt_str_lengths(100)
%matplotlib inline

def savefig_and_show(
    filename,
    plotdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/PhaseI/Plots",
):
    plt.savefig(f"{plotdir}/{filename}.pdf", bbox_inches="tight", dpi=300)

    os.system(f"code -r {plotdir}/{filename}.pdf")

def ggsavefig_and_show(
    plot,
    filename: str,
    plotdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/PhaseI/Plots",
    dpi=300,
):
    """
    plot: plotnine.ggplot.ggplot
    """
    plot.save(f"{plotdir}/{filename}.pdf", dpi=dpi)

    os.system(f"code -r {plotdir}/{filename}.pdf")

def chrom_assignment_using_paf(dsa_paf: str) -> pd.DataFrame:
    """Read a PAF file and return per-query primary chromosome assignment with alignment percentages."""
    paf_header = [
        "query_name", "q_len", "q_start", "q_end", "strand",
        "target_name", "t_len", "t_start", "t_end",
        "n_match", "block_len", "mapq", "id", "cigar",
    ] # NOTE: *.trimmed_paf doesn't have this header so you need to specify this

    df_paf_raw = pd.read_table(dsa_paf, sep="\t", header=None, names=paf_header)

    df_paf_raw["aligned_length"] = df_paf_raw["q_end"] - df_paf_raw["q_start"]
    agg = (
        df_paf_raw.groupby(["query_name", "target_name"])["aligned_length"]
        .sum()
        .reset_index()
    )
    max_chr = agg.loc[agg.groupby("query_name")["aligned_length"].idxmax()].copy()
    max_chr = max_chr.rename(
        columns={"target_name": "primary_chromosome", "aligned_length": "primary_aligned_length"}
    )
    total_aligned = agg.groupby("query_name")["aligned_length"].sum().reset_index()
    total_aligned = total_aligned.rename(columns={"aligned_length": "total_aligned_length"})
    query_lengths = df_paf_raw.drop_duplicates("query_name")[["query_name", "q_len"]]

    df_paf = max_chr.merge(total_aligned, on="query_name").merge(query_lengths, on="query_name")
    df_paf["primary_pct"] = df_paf["primary_aligned_length"] / df_paf["q_len"] * 100
    df_paf["other_pct"] = (
        (df_paf["total_aligned_length"] - df_paf["primary_aligned_length"])
        / df_paf["q_len"] * 100
    )
    return df_paf

# %%
# INFO: Comparing CDR positioning in COLO829 Passage A&B (to see if there is any evidence of CDR shift during passaging)
colo829_tb_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/CG_Methylation/Analysis/CDR/COLO829TB_DSA/COLO829TB_DSA.CDR.live.bed.gz")
colo829_tb_cdr["COLO829TB_Length"] = colo829_tb_cdr.lengths()

colo829_ta_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/CG_Methylation/Analysis/CDR/COLO829TA_DSA/COLO829TA_DSA.CDR.live.bed.gz")
colo829_ta_cdr["COLO829TA_Length"] = colo829_ta_cdr.lengths()

colo829_tab_cdr_length = pd.concat(
    [colo829_ta_cdr.groupby("Chromosome")["COLO829TA_Length"].sum(), 
     colo829_tb_cdr.groupby("Chromosome")["COLO829TB_Length"].sum()], axis=1).dropna(axis=0)

plot = (
    ggplot(colo829_tab_cdr_length, aes(x="COLO829TA_Length", y="COLO829TB_Length"))
    + geom_point()
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="COLO829 (Passage A) CDR Length (bp)",
        y="COLO829 (Passage B) CDR Length (bp)",
        title="CDR Length Comparison: COLO829TA vs COLO829TB",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_colo829_passages")

# %%
slope, intercept, r_value, p_value, std_err = linregress(
    colo829_tab_cdr_length["COLO829TA_Length"],
    colo829_tab_cdr_length["COLO829TB_Length"]
)
r_squared = r_value**2

plot = (
    ggplot(colo829_tab_cdr_length, aes(x="COLO829TA_Length", y="COLO829TB_Length"))
    + geom_point()
    + geom_smooth(method="lm", color="black", se=True)
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + annotate(
        "text",
        x=colo829_tab_cdr_length["COLO829TA_Length"].max() * 0.05,
        y=colo829_tab_cdr_length["COLO829TB_Length"].max() * 0.95,
        label=f"R² = {r_squared:.3f}",
        ha="left", size=11
    )
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="COLO829 (Passage A) CDR Length (bp)",
        y="COLO829 (Passage B) CDR Length (bp)",
        title="CDR Length Comparison: COLO829TA vs COLO829TB",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_colo829_passages_with_r2")

# %%
# INFO: Comparing CDR position between Passage A&B using Center-of-mass 
colo829_tb_cdr["midpoint"] = (colo829_tb_cdr["Start"] + colo829_tb_cdr["End"]) / 2
colo829_tb_cdr["weighted_mid"] = colo829_tb_cdr["midpoint"] * colo829_tb_cdr["COLO829TB_Length"]

colo829_tb_cdr_com = (
    colo829_tb_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / colo829_tb_cdr.groupby("Chromosome")["COLO829TB_Length"].sum()
)

colo829_ta_cdr["midpoint"] = (colo829_ta_cdr["Start"] + colo829_ta_cdr["End"]) / 2
colo829_ta_cdr["weighted_mid"] = colo829_ta_cdr["midpoint"] * colo829_ta_cdr["COLO829TA_Length"]

colo829_ta_cdr_com = (
    colo829_ta_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / colo829_ta_cdr.groupby("Chromosome")["COLO829TA_Length"].sum()
)

colo829_tab_cdr_com = pd.concat([colo829_tb_cdr_com, colo829_ta_cdr_com], axis=1, keys=['COLO829TB', 'COLO829TA']).dropna()
colo829_tab_cdr_com["CoM_Diff"] = abs(colo829_tab_cdr_com["COLO829TB"] - colo829_tab_cdr_com["COLO829TA"])

plot = (
    ggplot(colo829_tab_cdr_com, aes(x="''", y="CoM_Diff"))
    + geom_jitter(width=0.2, height=0)
    + labs(x="", y="Center of Mass Difference (bp)")
    + coord_cartesian(ylim=(0, 500000))
    + scale_y_continuous(labels=comma_format())
    + theme_minimal()
    + theme(
        figure_size=(4, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
    )
)

ggsavefig_and_show(plot, f"CoM_difference_in_cdr_colo829_passages")


fig, ax = plt.subplots(figsize=(4, 7))
sns.swarmplot(y=colo829_tab_cdr_com["CoM_Diff"], color="orange", ax=ax)
ax.set_ylabel("Center of Mass Difference (bp)")
ax.set_ylim(-50, 500000)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

savefig_and_show(f"CoM_difference_in_cdr_colo829_passages_swarmplot")


# %%
# INFO: Summary stats from Google BEST analysis for CDR-overlapping reads only
colo829bl_dsa_cdr_reads_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Revision/CDR_reads_to_other_asm/reads_overlapping_cdrs_from_COLO829BL_DSA_resetmapq.summary_identity_stats.csv")
colo829bl_dsa_cdr_reads_best_identity = colo829bl_dsa_cdr_reads_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

colo829bl_chm13_cdr_reads_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Revision/CDR_reads_to_other_asm/chm13/reads_overlapping_cdrs_from_COLO829BL_DSA_resetmapq_chm13.summary_identity_stats.csv")
colo829bl_chm13_cdr_reads_best_identity = colo829bl_chm13_cdr_reads_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

colo829bl_hg38_cdr_reads_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Revision/CDR_reads_to_other_asm/hg38/reads_overlapping_cdrs_from_COLO829BL_DSA_resetmapq_hg38.summary_identity_stats.csv")
colo829bl_hg38_cdr_reads_best_identity = colo829bl_hg38_cdr_reads_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

colo829bl_dsa_cdr_reads_best_cigar = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Revision/CDR_reads_to_other_asm/reads_overlapping_cdrs_from_COLO829BL_DSA_resetmapq.summary_cigar_stats.csv")
colo829bl_chm13_cdr_reads_best_cigar = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Revision/CDR_reads_to_other_asm/chm13/reads_overlapping_cdrs_from_COLO829BL_DSA_resetmapq_chm13.summary_cigar_stats.csv")

# %%
# INFO: Summary stats from Google BEST analysis for COLO829BL Fiber-seq reads (worth of 2 SMRT cells) aligned to DSA, T2T-CHM13v2.0, and hg38
colo829bl_dsa_two_rg_best_identity=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg.summary_identity_stats.csv")
colo829bl_dsa_two_rg_best_identity = colo829bl_dsa_two_rg_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

colo829bl_chm13_two_rg_best_identity=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13.summary_identity_stats.csv")
colo829bl_chm13_two_rg_best_identity = colo829bl_chm13_two_rg_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_hg38_two_rg_best_identity=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38.summary_identity_stats.csv")
colo829bl_hg38_two_rg_best_identity = colo829bl_hg38_two_rg_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

colo829bl_dsa_two_rg_best_cigar=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg.summary_cigar_stats.csv")
colo829bl_chm13_two_rg_best_cigar=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13.summary_cigar_stats.csv")
colo829bl_hg38_two_rg_best_cigar=pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38.summary_cigar_stats.csv")

# %%
# INFO: Identity stats in different genomic regions for DSA, GRCh38 and T2T-CHM13v2.0
# INFO: Centromeres
colo829bl_dsa_two_rg_centromere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg_centromere.summary_identity_stats.csv")
colo829bl_dsa_two_rg_centromere_best_identity = colo829bl_dsa_two_rg_centromere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_chm13_two_rg_centromere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13_centromere.summary_identity_stats.csv")
colo829bl_chm13_two_rg_centromere_best_identity = colo829bl_chm13_two_rg_centromere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_hg38_two_rg_centromere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38_centromere.summary_identity_stats.csv")
colo829bl_hg38_two_rg_centromere_best_identity = colo829bl_hg38_two_rg_centromere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

# INFO: Subtelomeres
colo829bl_dsa_two_rg_subtelomere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg_subtelomere.summary_identity_stats.csv")
colo829bl_dsa_two_rg_subtelomere_best_identity = colo829bl_dsa_two_rg_subtelomere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_chm13_two_rg_subtelomere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13_subtelomere.summary_identity_stats.csv")
colo829bl_chm13_two_rg_subtelomere_best_identity = colo829bl_chm13_two_rg_subtelomere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_hg38_two_rg_subtelomere_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38_subtelomere.summary_identity_stats.csv")
colo829bl_hg38_two_rg_subtelomere_best_identity = colo829bl_hg38_two_rg_subtelomere_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

# INFO: Segmentally Duplicated Regions
colo829bl_dsa_two_rg_segdup_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg_segdup.summary_identity_stats.csv")
colo829bl_dsa_two_rg_segdup_best_identity = colo829bl_dsa_two_rg_segdup_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_chm13_two_rg_segdup_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13_segdup.summary_identity_stats.csv")
colo829bl_chm13_two_rg_segdup_best_identity = colo829bl_chm13_two_rg_segdup_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_hg38_two_rg_segdup_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38_segdup.summary_identity_stats.csv")
colo829bl_hg38_two_rg_segdup_best_identity = colo829bl_hg38_two_rg_segdup_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

# INFO: Satellite Regions in general
colo829bl_dsa_two_rg_satellite_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/best/COLO829BL_DSA_resetmapq.calmd.two-rg_satellite.summary_identity_stats.csv")
colo829bl_dsa_two_rg_satellite_best_identity = colo829bl_dsa_two_rg_satellite_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_chm13_two_rg_satellite_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/chm13/COLO829BL_DSA_resetmapq.calmd.two-rg_chm13_satellite.summary_identity_stats.csv")
colo829bl_chm13_two_rg_satellite_best_identity = colo829bl_chm13_two_rg_satellite_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)
colo829bl_hg38_two_rg_satellite_best_identity = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/COLO829BL/hg38/COLO829BL_DSA_resetmapq.calmd.two-rg_hg38_satellite.summary_identity_stats.csv")
colo829bl_hg38_two_rg_satellite_best_identity = colo829bl_hg38_two_rg_satellite_best_identity.with_columns(
    (-10 * (1 - pl.col("gap_compressed_identity")).log(base=10)).alias("gap_compressed_identity_qv")
)

# %%
colo829bl_dsa_cdr_reads_best_cigar_cumsum = colo829bl_dsa_cdr_reads_best_cigar.sort(["cigar", "length"]).with_columns(
    pl.col("length_count_per_cigar")
    .cum_sum()
    .over("cigar")
    .alias("cumulative_proportion"))

plot = (
    ggplot(colo829bl_dsa_cdr_reads_best_cigar_cumsum, aes(x="length", y="cumulative_proportion"))
    + geom_line()
    + facet_wrap("~cigar", scales="free_x")
    + scale_x_continuous(labels=comma_format())
    + labs(
        x="CIGAR Operation Length (bp)",
        y="Cumulative Proportion",
        title="Cumulative Distribution of CIGAR Operation Lengths (CDR Reads in DSA)",
    )
    + theme_minimal()
    + theme(
        figure_size=(15, 5),
        text=element_text(family="Arial"), 
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"best_cigar_cumsum_colo829bl_dsa_cdr_reads")

colo829bl_chm13_cdr_reads_best_cigar_cumsum = colo829bl_chm13_cdr_reads_best_cigar.sort(["cigar", "length"]).with_columns(
    pl.col("length_count_per_cigar")
    .cum_sum()
    .over("cigar")
    .alias("cumulative_proportion"))

plot = (
    ggplot(colo829bl_chm13_cdr_reads_best_cigar_cumsum, aes(x="length", y="cumulative_proportion"))
    + geom_line()
    + facet_wrap("~cigar", scales="free_x")
    + scale_x_continuous(labels=comma_format())
    + labs(
        x="CIGAR Operation Length (bp)",
        y="Cumulative Proportion",
        title="Cumulative Distribution of CIGAR Operation Lengths (CDR Reads → CHM13)",
    )
    + theme_minimal()
    + theme(
        figure_size=(15, 5),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"best_cigar_cumsum_colo829bl_chm13_cdr_reads")
# %%
# INFO: Summary Identity Stats from BEST analysis
colo829bl_dsa_cdr_reads_best_identity_labeled = (
    colo829bl_dsa_cdr_reads_best_identity
    .with_columns(pl.lit("DSA Alignment").alias("Alignment"))
)
colo829bl_chm13_cdr_reads_best_identity_labeled = (
    colo829bl_chm13_cdr_reads_best_identity
    .with_columns(pl.lit("T2T-CHM13 Alignment").alias("Alignment"))
)

colo829bl_hg38_cdr_reads_best_identity_labeled = (
    colo829bl_hg38_cdr_reads_best_identity
    .with_columns(pl.lit("GRCh38 Alignment").alias("Alignment"))
)

combined = pl.concat([
    colo829bl_dsa_cdr_reads_best_identity_labeled,
    colo829bl_chm13_cdr_reads_best_identity_labeled,
    colo829bl_hg38_cdr_reads_best_identity_labeled,
])

combined_long = combined.unpivot(
    index="Alignment",
    on=[c for c in combined.columns if c not in ["total_alns", "primary_alns", "Alignment"]],
    variable_name="metric",
    value_name="value",
)

identity_metric_order = [
    "identity",
    "identity_qv",
    "gap_compressed_identity",
    "gap_compressed_identity_qv",
    "matches_per_kbp",
    "mismatches_per_kbp",
    "non_hp_ins_per_kbp",
    "non_hp_del_per_kbp",
    "hp_ins_per_kbp",
    "hp_del_per_kbp",
]

combined_long = combined_long.with_columns(
    pl.col("metric").cast(pl.Enum(identity_metric_order))
)

rename_map = {
    "identity": "BLAST Identity",
    "identity_qv": "Phred-scale BLAST Identity",
    "gap_compressed_identity": "Gap-compressed Identity",
    "gap_compressed_identity_qv": "Phred-scale Gap-compressed Identity",
    "matches_per_kbp": "Matches per Kbp",
    "mismatches_per_kbp": "Mismatches per Kbp",
    "non_hp_ins_per_kbp": "Non-Homopolymer Insertions per Kbp",
    "non_hp_del_per_kbp": "Non-Homopolymer Deletions per Kbp",
    "hp_ins_per_kbp": "Homopolymer Insertions per Kbp",
    "hp_del_per_kbp": "Homopolymer Deletions per Kbp",
}

combined_long = combined_long.with_columns(
    pl.col("metric")
    .cast(pl.String)
    .replace(rename_map)
    .cast(pl.Enum(list(rename_map.values())))
)

plot = (
    ggplot(combined_long, aes(x="Alignment", y="value", fill="Alignment"))
    + geom_col(width=0.6)
    + geom_text(aes(label="value"), va="bottom", size=7, format_string="{:.4f}")
    + facet_wrap("~metric", scales="free_y", ncol=4)
    + scale_y_continuous(labels=comma_format())
    + scale_fill_manual(values={"DSA Alignment": "#8b008b", "T2T-CHM13 Alignment": "#0054b4", "GRCh38 Alignment": "#228b22"})
    + labs(x="", y="Value", fill="Alignment", title="CDR Reads Alignment Quality (DSA vs T2T-CHM13 vs GRCh38)")
    + theme_minimal()
    + theme(
        figure_size=(12, 10),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black", rotation=20, ha="center"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
        panel_spacing=0.02
    )
)

ggsavefig_and_show(plot, "identity_stats_cdr_reads_dsa_vs_chm13_vs_hg38_barplot")

# %%
# INFO: Side-by-side comparison — All alignments + 4 region subsets, for DSA, T2T-CHM13v2.0, and GRCh38

def label_subset(df, assembly, region):
    return df.with_columns(
        pl.lit(assembly).alias("Assembly"),
        pl.lit(region).alias("Region"),
    )

region_order = [
    "All Alignments",
    "Subtelomeres",
    "Segmental Duplications",
    "Centromeres",
    "Satellite Regions",
]

subsets = [
    (colo829bl_dsa_two_rg_best_identity, colo829bl_chm13_two_rg_best_identity,             colo829bl_hg38_two_rg_best_identity,             "All Alignments"),
    (colo829bl_dsa_two_rg_subtelomere_best_identity, colo829bl_chm13_two_rg_subtelomere_best_identity, colo829bl_hg38_two_rg_subtelomere_best_identity, "Subtelomeres"),
    (colo829bl_dsa_two_rg_segdup_best_identity,      colo829bl_chm13_two_rg_segdup_best_identity,      colo829bl_hg38_two_rg_segdup_best_identity,      "Segmental Duplications"),
    (colo829bl_dsa_two_rg_centromere_best_identity,  colo829bl_chm13_two_rg_centromere_best_identity,  colo829bl_hg38_two_rg_centromere_best_identity,  "Centromeres"),
    (colo829bl_dsa_two_rg_satellite_best_identity,   colo829bl_chm13_two_rg_satellite_best_identity,   colo829bl_hg38_two_rg_satellite_best_identity,   "Satellite Regions"),
]

labeled = []
for dsa_df, chm13_df, hg38_df, region in subsets:
    labeled.append(label_subset(dsa_df, "DSA", region))
    labeled.append(label_subset(chm13_df, "T2T-CHM13", region))
    labeled.append(label_subset(hg38_df,  "GRCh38",   region))

combined = pl.concat(labeled)

combined_long = combined.unpivot(
    index=["Assembly", "Region"],
    on=[c for c in combined.columns
        if c not in ["total_alns", "primary_alns", "Assembly", "Region"]],
    variable_name="metric",
    value_name="value",
)

identity_metric_order = [
    "identity", "identity_qv",
    "gap_compressed_identity", "gap_compressed_identity_qv",
    "matches_per_kbp", "mismatches_per_kbp",
    "non_hp_ins_per_kbp", "non_hp_del_per_kbp",
    "hp_ins_per_kbp", "hp_del_per_kbp",
]
rename_map = {
    "identity": "BLAST Identity",
    "identity_qv": "Phred-scale BLAST Identity",
    "gap_compressed_identity": "Gap-compressed Identity",
    "gap_compressed_identity_qv": "Phred-scale Gap-compressed Identity",
    "matches_per_kbp": "Matches per Kbp",
    "mismatches_per_kbp": "Mismatches per Kbp",
    "non_hp_ins_per_kbp": "Non-Homopolymer Insertions per Kbp",
    "non_hp_del_per_kbp": "Non-Homopolymer Deletions per Kbp",
    "hp_ins_per_kbp": "Homopolymer Insertions per Kbp",
    "hp_del_per_kbp": "Homopolymer Deletions per Kbp",
}

combined_long = (
    combined_long
    .with_columns(pl.col("metric").cast(pl.Enum(identity_metric_order)))
    .with_columns(
        pl.col("metric").cast(pl.String).replace(rename_map).cast(pl.Enum(list(rename_map.values()))),
        pl.col("Region").cast(pl.Enum(region_order)),
    )
    .with_columns(
        (pl.col("Assembly") + " (" + pl.col("Region").cast(pl.String) + ")").alias("Group")
    )
)

dsa_reds = ["#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d"]
chm13_blues  = ["#bdd7e7", "#6baed6", "#3182bd", "#08519c", "#08306b"]
hg38_greens  = ["#bae4b3", "#74c476", "#31a354", "#006d2c", "#00441b"]

palette = {}
for i, region in enumerate(region_order):
    palette[f"DSA ({region})"] = dsa_reds[i]
    palette[f"T2T-CHM13 ({region})"] = chm13_blues[i]
    palette[f"GRCh38 ({region})"]    = hg38_greens[i]

group_order = (
    [f"DSA ({r})" for r in region_order]
    + [f"T2T-CHM13 ({r})" for r in region_order]
    + [f"GRCh38 ({r})"   for r in region_order]
)
combined_long = combined_long.with_columns(pl.col("Group").cast(pl.Enum(group_order)))

plot = (
    ggplot(combined_long, aes(x="Region", y="value", fill="Group"))
    + geom_col(position=position_dodge(preserve="single", width=0.85), width=0.8)
    + geom_text(
    aes(label="value"),
    position=position_dodge(width=0.85),
    va="bottom",
    size=5,
    format_string="{:.3f}",
    )
    + facet_wrap("~metric", scales="free_y", ncol=4)
    + scale_y_continuous(labels=comma_format())
    + scale_fill_manual(values=palette)
    + labs(
        x="",
        y="Value",
        fill="Assembly (Region)",
        title="COLO829BL Fiber-seq Alignment Quality across Genomic Regions",
    )
    + theme_minimal()
    + theme(
        figure_size=(16, 11),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black", rotation=30, ha="right"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
        panel_spacing=0.04,
        legend_position="right",
    )
)

ggsavefig_and_show(plot, "identity_stats_two_rg_all_vs_regions_dsa_vs_chm13_vs_hg38_barplot")

# %%
# INFO: Summary Identity Stats from BEST analysis for all reads (Two SMRT cells worth of sequencing data aligned to DSA, T2T-CHM13v2.0, and hg38)
colo829bl_dsa_two_rg_best_identity_labeled = (
    colo829bl_dsa_two_rg_best_identity
    .with_columns(pl.lit("DSA Alignment").alias("Alignment"))
)

colo829bl_chm13_two_rg_best_identity_labeled = (
    colo829bl_chm13_two_rg_best_identity
    .with_columns(pl.lit("T2T-CHM13 Alignment").alias("Alignment"))
)

colo829bl_hg38_two_rg_best_identity_labeled = (
    colo829bl_hg38_two_rg_best_identity
    .with_columns(pl.lit("GRCh38 Alignment").alias("Alignment"))
)

combined = pl.concat([
    colo829bl_dsa_two_rg_best_identity_labeled,
    colo829bl_chm13_two_rg_best_identity_labeled,
    colo829bl_hg38_two_rg_best_identity_labeled,
])

combined_long = combined.unpivot(
    index="Alignment",
    on=[c for c in combined.columns if c not in ["total_alns", "primary_alns", "Alignment"]],
    variable_name="metric",
    value_name="value",
)

identity_metric_order = [
    "identity",
    "identity_qv",
    "gap_compressed_identity",
    "gap_compressed_identity_qv",
    "matches_per_kbp",
    "mismatches_per_kbp",
    "non_hp_ins_per_kbp",
    "non_hp_del_per_kbp",
    "hp_ins_per_kbp",
    "hp_del_per_kbp",
]

combined_long = combined_long.with_columns(
    pl.col("metric").cast(pl.Enum(identity_metric_order))
)

rename_map = {
    "identity": "BLAST Identity",
    "identity_qv": "Phred-scale BLAST Identity",
    "gap_compressed_identity": "Gap-compressed Identity",
    "gap_compressed_identity_qv": "Phred-scale Gap-compressed Identity",
    "matches_per_kbp": "Matches per Kbp",
    "mismatches_per_kbp": "Mismatches per Kbp",
    "non_hp_ins_per_kbp": "Non-Homopolymer Insertions per Kbp",
    "non_hp_del_per_kbp": "Non-Homopolymer Deletions per Kbp",
    "hp_ins_per_kbp": "Homopolymer Insertions per Kbp",
    "hp_del_per_kbp": "Homopolymer Deletions per Kbp",
}

combined_long = combined_long.with_columns(
    pl.col("metric")
    .cast(pl.String)
    .replace(rename_map)
    .cast(pl.Enum(list(rename_map.values())))
)

plot = (
    ggplot(combined_long, aes(x="Alignment", y="value", fill="Alignment"))
    + geom_col(width=0.6)
    + geom_text(aes(label="value"), va="bottom", size=7, format_string="{:.4f}")
    + facet_wrap("~metric", scales="free_y", ncol=4)
    + scale_y_continuous(labels=comma_format())
    + scale_fill_manual(values={"DSA Alignment": "#8b008b", "T2T-CHM13 Alignment": "#0054b4", "GRCh38 Alignment": "#228b22"})
    + labs(x="", y="Value", fill="Alignment", title="Alignment Quality (DSA vs T2T-CHM13 vs GRCh38)")
    + theme_minimal()
    + theme(
        figure_size=(12, 10),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black", rotation=20, ha="center"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
        panel_spacing=0.02
    )
)

ggsavefig_and_show(plot, "identity_stats_two_rg_dsa_vs_chm13_vs_hg38_barplot")


# %% [markdown]
# Assigning chromosome number for Benchmarking-donor-DSAs (ST001-004) based on PAF between DSA and T2T-CHM13v2.0 assembly

# %%
# %%
# INFO: Contig-Chromosome Assignment (using mapping between DSA and T2T-CHM13v2.0)
# NOTE: Written originally by Youngjun Kwon @Eichler lab and slightly modified by Sohny

pafdir="/mmfs1/gscratch/stergachislab/mhsohny/Tools/asm-to-reference-alignment/results/T2T_chm13/chain"
st001_paf = f"{pafdir}/DSA_ST001_1_2.cat.paf"
st002_paf = f"{pafdir}/DSA_ST002_1_2.cat.paf"
st003_paf = f"{pafdir}/DSA_ST003_1_2.cat.paf"
st004_paf = f"{pafdir}/DSA_ST004_1_2.cat.paf"

outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis"

chrom_assignment_using_paf(st001_paf).to_csv(
    f"{outdir}/DSA_ST001_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

chrom_assignment_using_paf(st002_paf).to_csv(
    f"{outdir}/DSA_ST002_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

chrom_assignment_using_paf(st003_paf).to_csv(
    f"{outdir}/DSA_ST003_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

chrom_assignment_using_paf(st004_paf).to_csv(
    f"{outdir}/DSA_ST004_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

# %%
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/SMHTLBLA2"

smhtlbla2_paf = f"{pafdir}/SMHTLBLA2-DSA_1_2.cat.paf"
chrom_assignment_using_paf(smhtlbla2_paf).to_csv(
    f"{outdir}/SMHTLBLA2-DSA_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)
# %%
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR2-PS01517_PS01518"

gm28572_gm28570_paf = f"{pafdir}/DONOR2-PS01517_PS01518_1_2.cat.paf"
chrom_assignment_using_paf(gm28572_gm28570_paf).to_csv(
    f"{outdir}/GM28572_GM28570_DSA_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

# %%
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR1-PS01519_PS01520"

gm25456_gm25455_paf = f"{pafdir}/DONOR1-PS01519_PS01520_1_2.cat.paf"
chrom_assignment_using_paf(gm25456_gm25455_paf).to_csv(
    f"{outdir}/GM25456_GM25455_DSA_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False
)

# %%
# INFO: Comparing CDR positioning in GM25456 (Fibroblast) and GM25455 (LCL)
gm25456_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR1-PS01519_PS01520/annotation/centromere/CDR/FromPacBio/GM25456_Fibroblast_Fiber-seq_merged.CDR.live.bed.gz")
gm25456_cdr["GM25456_Length"] = gm25456_cdr.lengths()

gm25455_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR1-PS01519_PS01520/annotation/centromere/CDR/FromPacBio/GM25455_LCL_Fiber-seq_merged.CDR.live.bed.gz")
gm25455_cdr["GM25455_Length"] = gm25455_cdr.lengths()

gm25456_gm25455_cdr_length = pd.concat(
    [gm25456_cdr.groupby("Chromosome")["GM25456_Length"].sum(), 
     gm25455_cdr.groupby("Chromosome")["GM25455_Length"].sum()], axis=1).dropna(axis=0)

plot = (
    ggplot(gm25456_gm25455_cdr_length, aes(x="GM25456_Length", y="GM25455_Length"))
    + geom_point()
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="GM25456 (Fibroblast) CDR Length (bp)",
        y="GM25455 (LCL) CDR Length (bp)",
        title="CDR Length Comparison: GM25456 vs GM25455",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_gm25456_gm25455")

# %%
slope, intercept, r_value, p_value, std_err = linregress(
    gm25456_gm25455_cdr_length["GM25456_Length"],
    gm25456_gm25455_cdr_length["GM25455_Length"]
)
r_squared = r_value**2

plot = (
    ggplot(gm25456_gm25455_cdr_length, aes(x="GM25456_Length", y="GM25455_Length"))
    + geom_point()
    + geom_smooth(method="lm", color="black", se=True)
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + annotate(
        "text",
        x=gm25456_gm25455_cdr_length["GM25456_Length"].max() * 0.05,
        y=gm25456_gm25455_cdr_length["GM25455_Length"].max() * 0.95,
        label=f"R² = {r_squared:.3f}",
        ha="left", size=11
    )
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="GM25456 (Fibroblast) CDR Length (bp)",
        y="GM25455 (LCL) CDR Length (bp)",
        title="CDR Length Comparison: GM25456 vs GM25455",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_gm25456_gm25455_with_r2")
# %%
# INFO: Comparing CDR position between Passage A&B using Center-of-mass 
gm25456_cdr["midpoint"] = (gm25456_cdr["Start"] + gm25456_cdr["End"]) / 2
gm25456_cdr["weighted_mid"] = gm25456_cdr["midpoint"] * gm25456_cdr["GM25456_Length"]

gm25456_cdr_com = (
    gm25456_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / gm25456_cdr.groupby("Chromosome")["GM25456_Length"].sum()
)

gm25455_cdr["midpoint"] = (gm25455_cdr["Start"] + gm25455_cdr["End"]) / 2
gm25455_cdr["weighted_mid"] = gm25455_cdr["midpoint"] * gm25455_cdr["GM25455_Length"]

gm25455_cdr_com = (
    gm25455_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / gm25455_cdr.groupby("Chromosome")["GM25455_Length"].sum()
)

gm25456_gm25455_cdr_com = pd.concat([gm25456_cdr_com, gm25455_cdr_com], axis=1, keys=['GM25456', 'GM25455']).dropna()
gm25456_gm25455_cdr_com["CoM_Diff"] = abs(gm25456_gm25455_cdr_com["GM25456"] - gm25456_gm25455_cdr_com["GM25455"])

plot = (
    ggplot(gm25456_gm25455_cdr_com, aes(x="''", y="CoM_Diff"))
    + geom_jitter(width=0.2, height=0)
    + labs(x="", y="Center of Mass Difference (bp)")
    + coord_cartesian(ylim=(0, 500000))
    + scale_y_continuous(labels=comma_format())
    + theme_minimal()
    + theme(
        figure_size=(4, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
    )
)

ggsavefig_and_show(plot, f"CoM_difference_in_cdr_gm25456_gm25455")


fig, ax = plt.subplots(figsize=(4, 7))
sns.swarmplot(y=gm25456_gm25455_cdr_com["CoM_Diff"], color="orange", ax=ax)
ax.set_ylabel("Center of Mass Difference (bp)")
ax.set_ylim(-50, 500000)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

savefig_and_show(f"CoM_difference_in_cdr_gm25456_gm25455_swarmplot")
# %%
# INFO: Comparing CDR positioning in GM28572 (Fibroblast) and GM28570 (LCL)
gm28572_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR2-PS01517_PS01518/annotation/centromere/CDR/FromPacBio/GM28572_Fibroblast_Fiber-seq_merged.CDR.live.bed.gz")
gm28572_cdr["GM28572_Length"] = gm28572_cdr.lengths()

gm28570_cdr = pr.read_bed("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Fibroblast-LCL/DONOR2-PS01517_PS01518/annotation/centromere/CDR/FromPacBio/GM28570_LCL_Fiber-seq_merged.CDR.live.bed.gz")
gm28570_cdr["GM28570_Length"] = gm28570_cdr.lengths()

gm28572_gm28570_cdr_length = pd.concat(
    [gm28572_cdr.groupby("Chromosome")["GM28572_Length"].sum(), 
     gm28570_cdr.groupby("Chromosome")["GM28570_Length"].sum()], axis=1).dropna(axis=0)

plot = (
    ggplot(gm28572_gm28570_cdr_length, aes(x="GM28572_Length", y="GM28570_Length"))
    + geom_point()
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="GM28572 (Fibroblast) CDR Length (bp)",
        y="GM28570 (LCL) CDR Length (bp)",
        title="CDR Length Comparison: GM28572 vs GM28570",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_gm28572_gm28570")

# %%
slope, intercept, r_value, p_value, std_err = linregress(
    gm28572_gm28570_cdr_length["GM28572_Length"],
    gm28572_gm28570_cdr_length["GM28570_Length"]
)
r_squared = r_value**2

plot = (
    ggplot(gm28572_gm28570_cdr_length, aes(x="GM28572_Length", y="GM28570_Length"))
    + geom_point()
    + geom_smooth(method="lm", color="black", se=True)
    + geom_abline(intercept=0, slope=1, linetype="dotted", color="gray")
    + annotate(
        "text",
        x=gm28572_gm28570_cdr_length["GM28572_Length"].max() * 0.05,
        y=gm28572_gm28570_cdr_length["GM28570_Length"].max() * 0.95,
        label=f"R² = {r_squared:.3f}",
        ha="left", size=11
    )
    + scale_x_continuous(labels=comma_format())
    + scale_y_continuous(labels=comma_format())
    + labs(
        x="GM28572 (Fibroblast) CDR Length (bp)",
        y="GM28570 (LCL) CDR Length (bp)",
        title="CDR Length Comparison: GM28572 vs GM28570",
    )
    + theme_minimal()
    + theme(
        figure_size=(7, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
        axis_ticks_minor_x=element_line(size=0.5, color="darkgray"),
        axis_ticks_minor_y=element_line(size=0.5, color="darkgray"),
    )
)

ggsavefig_and_show(plot, f"length_comparison_cdr_gm28572_gm28570_with_r2")
# %%
# INFO: Comparing CDR position between Passage A&B using Center-of-mass 
gm28572_cdr["midpoint"] = (gm28572_cdr["Start"] + gm28572_cdr["End"]) / 2
gm28572_cdr["weighted_mid"] = gm28572_cdr["midpoint"] * gm28572_cdr["GM28572_Length"]

gm28572_cdr_com = (
    gm28572_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / gm28572_cdr.groupby("Chromosome")["GM28572_Length"].sum()
)

gm28570_cdr["midpoint"] = (gm28570_cdr["Start"] + gm28570_cdr["End"]) / 2
gm28570_cdr["weighted_mid"] = gm28570_cdr["midpoint"] * gm28570_cdr["GM28570_Length"]

gm28570_cdr_com = (
    gm28570_cdr.groupby("Chromosome")["weighted_mid"].sum()
    / gm28570_cdr.groupby("Chromosome")["GM28570_Length"].sum()
)

gm28572_gm28570_cdr_com = pd.concat([gm28572_cdr_com, gm28570_cdr_com], axis=1, keys=['GM28572', 'GM28570']).dropna()
gm28572_gm28570_cdr_com["CoM_Diff"] = abs(gm28572_gm28570_cdr_com["GM28572"] - gm28572_gm28570_cdr_com["GM28570"])

plot = (
    ggplot(gm28572_gm28570_cdr_com, aes(x="''", y="CoM_Diff"))
    + geom_jitter(width=0.2, height=0)
    + labs(x="", y="Center of Mass Difference (bp)")
    + coord_cartesian(ylim=(0, 500000))
    + scale_y_continuous(labels=comma_format())
    + theme_minimal()
    + theme(
        figure_size=(4, 7),
        text=element_text(family="Arial"),
        axis_text_x=element_text(color="black"),
        axis_text_y=element_text(color="black"),
        axis_line_x=element_line(size=0.5, color="black"),
        axis_line_y=element_line(size=0.5, color="black"),
        axis_ticks_major=element_line(size=0.5, color="black"),
    )
)

ggsavefig_and_show(plot, f"CoM_difference_in_cdr_gm28572_gm28570")


fig, ax = plt.subplots(figsize=(4, 7))
sns.swarmplot(y=gm28572_gm28570_cdr_com["CoM_Diff"], color="orange", ax=ax)
ax.set_ylabel("Center of Mass Difference (bp)")
ax.set_ylim(-50, 500000)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

savefig_and_show(f"CoM_difference_in_cdr_gm28572_gm28570_swarmplot")

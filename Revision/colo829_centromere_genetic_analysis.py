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

def read_vcf(path):
    if path[-3:] == ".gz":
        with gz.open(path, "rb") as f:
            lines = [l.decode("utf-8") for l in f if not l.startswith(b"##")]
            return pd.read_csv(
                io.StringIO("".join(lines)),
                dtype={
                    "#CHROM": str,
                    "POS": int,
                    "ID": str,
                    "REF": str,
                    "ALT": str,
                    "QUAL": str,
                    "FILTER": str,
                    "INFO": str,
                },
                sep="\t",
            ).rename(columns={"#CHROM": "CHROM"})
    else:
        with open(path, "r") as f:
            lines = [l for l in f if not l.startswith("##")]
            return pd.read_csv(
                io.StringIO("".join(lines)),
                dtype={
                    "#CHROM": str,
                    "POS": int,
                    "ID": str,
                    "REF": str,
                    "ALT": str,
                    "QUAL": str,
                    "FILTER": str,
                    "INFO": str,
                },
                sep="\t",
            ).rename(columns={"#CHROM": "CHROM"})


# %%
def get_sv_table_from_scanCSV(
    scancsv_output: pl.DataFrame,
    ins_or_del: str | None = None,
    min_sv_length: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    scancsv_output: polars DataFrame from scanCSV output
    ins_or_del: "INS" or "DEL" or None (no type filter)

    Returns (sv_df, counts_table):
        sv_df: filtered SVs
        counts_table: zero-filled indel_length value counts (for autocorrelation)
    """
    sv = scancsv_output.filter(pl.col("indel_length") >= min_sv_length)
    if ins_or_del is not None:
        sv = sv.filter(pl.col("indel_type") == ins_or_del)

    counts = sv["indel_length"].value_counts().sort("indel_length")
    """
    value_range = pl.DataFrame({
        "indel_length": range(sv["indel_length"].min(), sv["indel_length"].max() + 1)
    })
    """
    value_range = pl.DataFrame({
        "indel_length": range(0, sv["indel_length"].max() + 1)
    })
    counts_table = (
        value_range
        .join(counts, on="indel_length", how="left")
        .with_columns(pl.col("count").fill_null(0))
    )

    return sv, counts_table



# %%
def collapse_overlapping_sv(df: pl.DataFrame) -> pl.DataFrame:
    group_keys = ["#chrom", "indel_type", "overlap_class"]

    result = (
        df
        .with_columns(
            (pl.col("start") - pl.col("indel_length")).alias("padded_start"),
            (pl.col("end") + pl.col("indel_length")).alias("padded_end"),
            pl.col("unit_multiple").round(0).alias("rounded_unit_multiple"),
        )
        .sort(group_keys + ["padded_start"])
        .with_columns(
            pl.col("padded_end")
            .cum_max()
            .shift(1)
            .over(group_keys + ["rounded_unit_multiple"])
            .alias("_prev_end_max")
        )
        .with_columns(
            ((pl.col("padded_start") > pl.col("_prev_end_max")) | pl.col("_prev_end_max").is_null())
            .cast(pl.Int32)
            .cum_sum()
            .over(group_keys + ["rounded_unit_multiple"])
            .alias("_cluster_id")
        )
        .unique(subset=group_keys + ["rounded_unit_multiple", "_cluster_id", "read_name"], keep="first")
        .with_columns(
            pl.len().over(group_keys + ["rounded_unit_multiple", "_cluster_id"]).alias("n_collapsed")
        )
        .group_by(group_keys + ["rounded_unit_multiple", "_cluster_id"], maintain_order=True)
        .first()
        .with_columns(
            pl.col("padded_start").alias("start"),
            pl.col("padded_end").alias("end"),
        )
        .drop(["_prev_end_max", "_cluster_id", "rounded_unit_multiple", "padded_start", "padded_end"])
    )

    return result

# %%
def _assign_size_subcluster_ids(group_df: pl.DataFrame, size_ratio_threshold: float) -> pl.DataFrame:
    sub_id = 0
    current_min: float | None = None
    sub_ids = []
    for length in group_df["indel_length"]:
        if current_min is None or length > current_min / size_ratio_threshold:
            sub_id += 1
            current_min = float(length)
        sub_ids.append(sub_id)
    return group_df.with_columns(pl.Series(name="_size_subcluster_id", values=sub_ids, dtype=pl.Int32))

# %%
# INFO: New non-unit SV collapser
def collapse_overlapping_non_unit_sv(df: pl.DataFrame, size_ratio_threshold: float = 0.9) -> pl.DataFrame:
    group_keys = ["#chrom", "indel_type", "overlap_class"]

    after_pos_cluster = (
        df
        .with_columns(
            (pl.col("start") - pl.col("indel_length")).alias("padded_start"),
            (pl.col("end") + pl.col("indel_length")).alias("padded_end"),
        )
        .sort(group_keys + ["padded_start"])
        .with_columns(
            pl.col("padded_end")
            .cum_max()
            .shift(1)
            .over(group_keys)
            .alias("_prev_end_max")
        )
        .with_columns(
            ((pl.col("padded_start") > pl.col("_prev_end_max")) | pl.col("_prev_end_max").is_null())
            .cast(pl.Int32)
            .cum_sum()
            .over(group_keys)
            .alias("_pos_cluster_id")
        )
        .sort(group_keys + ["_pos_cluster_id", "indel_length"])
    )

    pieces = [
        _assign_size_subcluster_ids(g, size_ratio_threshold)
        for g in after_pos_cluster.partition_by(group_keys + ["_pos_cluster_id"], maintain_order=True)
    ]
    if pieces:
        after_size_cluster = pl.concat(pieces)
    else:
        after_size_cluster = after_pos_cluster.with_columns(pl.lit(0, dtype=pl.Int32).alias("_size_subcluster_id"))

    cluster_cols = group_keys + ["_pos_cluster_id", "_size_subcluster_id"]

    result = (
        after_size_cluster
        .sort(cluster_cols + ["padded_start"])
        .with_row_index(name="_row_idx")
        .with_columns(
            pl.col("padded_start")
            .rank(method="ordinal")
            .over(cluster_cols + ["read_name"])
            .alias("_read_rank")
        )
        .with_columns(
            (pl.col("_read_rank").max().over(cluster_cols) > 1).alias("has_intra_read_conflict")
        )
        .with_columns(
            pl.when(pl.col("_read_rank") > 1)
            .then(pl.format("exile-{}", pl.col("_row_idx")))
            .otherwise(pl.format(
                "main-{}-{}-{}-{}-{}",
                pl.col("#chrom"),
                pl.col("indel_type"),
                pl.col("overlap_class"),
                pl.col("_pos_cluster_id"),
                pl.col("_size_subcluster_id"),
            ))
            .alias("_final_cluster_id")
        )
        .with_columns(
            pl.len().over("_final_cluster_id").alias("n_collapsed")
        )
        .group_by("_final_cluster_id", maintain_order=True)
        .first()
        .with_columns(
            pl.col("padded_start").alias("start"),
            pl.col("padded_end").alias("end"),
        )
        .drop([
            "_prev_end_max",
            "_pos_cluster_id",
            "_size_subcluster_id",
            "_row_idx",
            "_read_rank",
            "_final_cluster_id",
            "padded_start",
            "padded_end",
        ])
    )

    return result

def autocorr_range(series: pl.Series, max_lag: int) -> pl.DataFrame:
    s = series.to_pandas()
    return pl.DataFrame({
        "lag": range(1, max_lag + 1),
        "autocorr": [s.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    })

# %%
# NOTE: 
# min_dist_query / read_length
# min_dist_ref / alignment_span
# %%
centroindel_colo_dir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/Structural_Variations/Centromeric_SV/"

colo829bl_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Centromere/COLO829BL_v3.0.0/DSA_COLO829BL_v3.0.0_1_2.trimmed_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

colo829bl_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Centromere/COLO829BL_v3.0.0/CDR/bulk_cdr_bl_r9_medconf_live.bed", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

#colo829bl_cdr = colo829bl_cdr.merge_overlaps(slack=1000000000) # FIXME: This needs to be amended. When we are calculating the reate of SNV in CDR vs Non-CDR we used 50kb padded CDR pyranges object. But when we are scanning for the unit-sized indels in centromere we made bunch of 1kb windows, and tag the windows that overlapped with the collapsed CDR like the above. This discrepancies should be fixed.

colo829bl_cdr = colo829bl_cdr.extend_ranges(50_000).merge_overlaps()

colo829bl_flagger_nucflag = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_Flagger-NucFlag.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
colo829bl_flagger_nucflag_100kb_del = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_Flagger-NucFlag_100kb-DEL_removed.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

# %%
# INFO: COLO829BL
#['#chrom','start','end','indel_id','indel_type','indel_length','unit_multiple','is_unit_sized','median_bq','overlap_class','gc_identity','aligned_fraction','read_length','alignment_span','min_dist_query','min_dist_ref','read_name','is_primary','sequence']
colobl_fiberseq_uid = pl.read_csv(f"{centroindel_colo_dir}/COLO829BL_Fiber-seq_tolerance0.95/COLO829BL_Fiber-seq_CDR_unitsized-indels_tolerance0.95.bed", separator="\t")
colotb_fiberseq_uid = pl.read_csv(f"{centroindel_colo_dir}/COLO829TB_Fiber-seq_tolerance0.95/COLO829TB_Fiber-seq_CDR_unitsized-indels_tolerance0.95.bed", separator="\t")

colobl_ulont_uid = pl.read_csv(f"{centroindel_colo_dir}/COLO829BL_UL-ONT_tolerance0.95/COLO829BL_UL-ONT_CDR_unitsized-indels_tolerance0.95.bed", separator="\t")
colotb_ulont_uid = pl.read_csv(f"{centroindel_colo_dir}/COLO829TB_UL-ONT_tolerance0.95/COLO829TB_UL-ONT_CDR_unitsized-indels_tolerance0.95.bed", separator="\t")

# NOTE: Total SVs
colobl_fiberseq_uid_sv, colobl_fiberseq_uid_sv_counts_table = get_sv_table_from_scanCSV(colobl_fiberseq_uid)
# NOTE: Insertion and Deletion SVs
colobl_fiberseq_uid_sv_ins, colobl_fiberseq_uid_sv_ins_counts_table = get_sv_table_from_scanCSV(colobl_fiberseq_uid, "INS")
colobl_fiberseq_uid_sv_del, colobl_fiberseq_uid_sv_del_counts_table = get_sv_table_from_scanCSV(colobl_fiberseq_uid, "DEL")

# INFO: COLO829TB
#['#chrom','start','end','indel_id','indel_type','indel_length','unit_multiple','is_unit_sized','median_bq','overlap_class','gc_identity','aligned_fraction','read_length','alignment_span','min_dist_query','min_dist_ref','read_name','is_primary','sequence']
# NOTE: Total SVs
colotb_fiberseq_uid_sv, colotb_fiberseq_uid_sv_counts_table = get_sv_table_from_scanCSV(colotb_fiberseq_uid)
# NOTE: Insertion and Deletion SVs
colotb_fiberseq_uid_sv_ins, colotb_fiberseq_uid_sv_ins_counts_table = get_sv_table_from_scanCSV(colotb_fiberseq_uid, "INS")
colotb_fiberseq_uid_sv_del, colotb_fiberseq_uid_sv_del_counts_table = get_sv_table_from_scanCSV(colotb_fiberseq_uid, "DEL")

# INFO: ##########################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
# INFO: ##########################################################################################

# %%
# INFO: Obtaining SV sets for COLO829BL and COLO829 
# INFO: COLO829BL Fiber-seq
colobl_fiberseq_uid_sv_pr = pr.PyRanges(
    colobl_fiberseq_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") - 1) # NOTE: For INS, start == end, so need to fix
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

colobl_fiberseq_uid_sv_pr_colo829bl_flagger_nucflag_overlap_idx = colobl_fiberseq_uid_sv_pr.overlap(colo829bl_flagger_nucflag)["_idx"].values

colobl_fiberseq_uid_sv_filtered = (colobl_fiberseq_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(colobl_fiberseq_uid_sv_pr_colo829bl_flagger_nucflag_overlap_idx))
    .drop("_idx"))

colobl_fiberseq_uid_sv_filtered_total = colobl_fiberseq_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

# INFO: COLO829TB Fiber-seq
colotb_fiberseq_uid_sv_pr = pr.PyRanges(
    colotb_fiberseq_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1) # NOTE: For INS, start == end, so need to fix
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

colotb_fiberseq_uid_sv_pr_colo829bl_flagger_nucflag_100kb_del_overlap_idx = colotb_fiberseq_uid_sv_pr.overlap(colo829bl_flagger_nucflag_100kb_del)["_idx"].values

colotb_fiberseq_uid_sv_filtered = (colotb_fiberseq_uid_sv
    .with_row_index("_idx")
    .filter(pl.col("_idx").is_in(colotb_fiberseq_uid_sv_pr_colo829bl_flagger_nucflag_100kb_del_overlap_idx))
    .drop("_idx")) # NOTE: Here, instead of filtering out SNVs that overlap, we are keeping only those that overlap with the 100 kb flagged regions.

colotb_fiberseq_uid_sv_filtered_total = colotb_fiberseq_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

# INFO: Obtaining the overallped SVs (Added)
colobl_colotb_overlapping_indel_id = set(colobl_fiberseq_uid_sv_filtered_total["indel_id"].to_list()) & set(colotb_fiberseq_uid_sv_filtered_total["indel_id"].to_list())

# INFO: Removing the overlapped SVs (Added)
# INFO: COLO829BL Fiber-seq
colobl_fiberseq_uid_sv_filtered_total = colobl_fiberseq_uid_sv_filtered_total.filter(
    ~pl.col("indel_id").is_in(colobl_colotb_overlapping_indel_id)
)

colobl_fiberseq_uid_sv_filtered_unit = colobl_fiberseq_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
    )

colobl_fiberseq_uid_sv_filtered_nonunit = colobl_fiberseq_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
    )

colobl_fiberseq_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(colobl_fiberseq_uid_sv_filtered_unit)
colobl_fiberseq_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(colobl_fiberseq_uid_sv_filtered_nonunit)

colobl_fiberseq_uid_sv_filtered_total_collapsed = pl.concat([colobl_fiberseq_uid_sv_filtered_unit_collapsed, colobl_fiberseq_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: COLO829BL a-satellite SV in CDR vs. Non-CDR (using Fiber-seq)
colobl_fiberseq_counts_table = colobl_fiberseq_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
colobl_fiberseq_unit_sv_cdr = colobl_fiberseq_counts_table.filter(colobl_fiberseq_counts_table["overlap_class"] == "target")["count"][0]
colobl_fiberseq_unit_sv_non_cdr = colobl_fiberseq_counts_table.filter(colobl_fiberseq_counts_table["overlap_class"] == "non_target")["count"][0]

colobl_fiberseq_contingency_table = [
    [colobl_fiberseq_unit_sv_cdr, colobl_fiberseq_unit_sv_non_cdr],
    [colo829bl_cdr.subtract_overlaps(colo829bl_flagger_nucflag).length, colo829bl_centromere.subtract_overlaps(colo829bl_flagger_nucflag).length - colo829bl_cdr.subtract_overlaps(colo829bl_flagger_nucflag).length]
    ]

colobl_fiberseq_res = chi2_contingency(colobl_fiberseq_contingency_table)

colobl_fiberseq_rate_cdr = colobl_fiberseq_unit_sv_cdr / colo829bl_cdr.subtract_overlaps(colo829bl_flagger_nucflag).length
colobl_fiberseq_rate_non_cdr = colobl_fiberseq_unit_sv_non_cdr / (colo829bl_centromere.subtract_overlaps(colo829bl_flagger_nucflag).length - 
colo829bl_cdr.subtract_overlaps(colo829bl_flagger_nucflag).length)

print(f"Chi-squared p-value: {colobl_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of COLO829BL:{colobl_fiberseq_rate_cdr}")
print(f"Rate of unit-length α-satellite SVs outside CDR of COLO829BL:{colobl_fiberseq_rate_non_cdr}")

# NOTE: Rate of unit-length α-satellite SVs within or outside CDR in COLO829BL
plot = (
    ggplot(
        pd.DataFrame({
            "region": ["CDR", "Non-CDR"],
            "rate":   [colobl_fiberseq_rate_cdr, colobl_fiberseq_rate_non_cdr],
        }),
        aes(x="region", y="rate", fill="region")
    ) +
    geom_col(width=0.7) +
    annotate("text", x=1.5, y=max(colobl_fiberseq_rate_cdr, colobl_fiberseq_rate_non_cdr) * 1.05,
             label=f"χ² p value = {colobl_fiberseq_res.pvalue:.2e}", size=9, color="black") +
    scale_fill_manual(values={"CDR": "#b89841", "Non-CDR": "#ac7daf"}) +
    scale_y_continuous(labels=scientific_format()) +
    labs(x="Region",
         y="Rate of unique unit-length α-satellite SVs (per bp)",
         title="COLO829BL (Fiber-seq, ~170X)") +
    theme_minimal() +
    theme(
        figure_size=(4.5, 4),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
        )
)

ggsavefig_and_show(plot, "rate_of_unit_length_alpha_satellite_svs_colo829bl_cdr_vs_non_cdr")

# %%
# INFO: Histogram of Total "unique" SV lengths
plot = (ggplot(colobl_fiberseq_uid_sv_filtered_total_collapsed, aes(x='indel_length')) +
        geom_histogram(breaks=np.linspace(0, 20_000, 1_000), alpha=0.7, size=0.5, color='black') +
        scale_x_continuous(labels=comma_format()) +
        scale_y_log10(labels=comma_format()) +
        labs(title=f"SV Length Histogram (COLO829BL Fiber-seq)", 
             x='SV Length (bp)', 
             y='Count (log 10)') +
        theme_minimal() +
        theme(figure_size=(15, 5),
              text=element_text(family='Arial'),
              plot_title=element_text(size=15, color='black', hjust=0.5),
              axis_text_x=element_text(size=10, color='black'),
              axis_text_y=element_text(size=10, color='black'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, f"sv_length_histogram_colo829bl_fiberseq_uid_sv")

# INFO: Histogram of SV lengths (zoomed in up to 5,000 bp)
plot = (ggplot(colobl_fiberseq_uid_sv_filtered_total_collapsed, aes(x='indel_length')) +
        geom_histogram(breaks=np.linspace(0, 5_000, 1_000), alpha=0.7, size=0.5, color='black') +
        scale_x_continuous(labels=comma_format()) +
        scale_y_log10(labels=comma_format()) +
        labs(title=f"SV Length Histogram (COLO829BL Fiber-seq), up to 5,000 bp only", 
             x='SV Length (bp)', 
             y='Count (log 10)') +
        theme_minimal() +
        theme(figure_size=(15, 5),
              text=element_text(family='Arial'),
              plot_title=element_text(size=15, color='black', hjust=0.5),
              axis_text_x=element_text(size=10, color='black'),
              axis_text_y=element_text(size=10, color='black'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, f"sv_length_histogram_colo829bl_fiberseq_uid_sv_zoomed_in_5000bp")

# %%
# INFO: Removing the overlapped SVs (Added)
# INFO: COLO829TB Fiber-seq
colotb_fiberseq_uid_sv_filtered_total = colotb_fiberseq_uid_sv_filtered_total.filter(
    ~pl.col("indel_id").is_in(colobl_colotb_overlapping_indel_id)
)

colotb_fiberseq_uid_sv_filtered_unit = colotb_fiberseq_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
    )

colotb_fiberseq_uid_sv_filtered_nonunit = colotb_fiberseq_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
    )

colotb_fiberseq_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(colotb_fiberseq_uid_sv_filtered_unit)
colotb_fiberseq_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(colotb_fiberseq_uid_sv_filtered_nonunit)

colotb_fiberseq_uid_sv_filtered_total_collapsed = pl.concat([colotb_fiberseq_uid_sv_filtered_unit_collapsed, colotb_fiberseq_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

colotb_fiberseq_counts_table = colotb_fiberseq_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
colotb_fiberseq_unit_sv_cdr = colotb_fiberseq_counts_table.filter(colotb_fiberseq_counts_table["overlap_class"] == "target")["count"][0]
colotb_fiberseq_unit_sv_non_cdr = colotb_fiberseq_counts_table.filter(colotb_fiberseq_counts_table["overlap_class"] == "non_target")["count"][0]


# INFO: CDR vs. Non-CDR for COLO829TB
colotb_fiberseq_contingency_table = [
    [colotb_fiberseq_unit_sv_cdr, colotb_fiberseq_unit_sv_non_cdr],
    [colo829bl_cdr.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length, colo829bl_centromere.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length - colo829bl_cdr.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length]
    ]

colotb_fiberseq_res = chi2_contingency(colotb_fiberseq_contingency_table)

colotb_fiberseq_rate_cdr = colotb_fiberseq_unit_sv_cdr / colo829bl_cdr.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length
colotb_fiberseq_rate_non_cdr = colotb_fiberseq_unit_sv_non_cdr / (colo829bl_centromere.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length - colo829bl_cdr.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del).length)

print(f"Chi-squared p-value: {colotb_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of COLO829:{colotb_fiberseq_rate_cdr}")
print(f"Rate of unit-length α-satellite SVs outside CDR of COLO829:{colotb_fiberseq_rate_non_cdr}")


# NOTE: Rate of unit-length α-satellite SVs within or outside CDR in COLO829
plot = (
    ggplot(
        pd.DataFrame({
            "region": ["CDR", "Non-CDR"],
            "rate":   [colotb_fiberseq_rate_cdr, colotb_fiberseq_rate_non_cdr],
        }),
        aes(x="region", y="rate", fill="region")
    ) +
    geom_col(width=0.7) +
    annotate("text", x=1.5, y=max(colotb_fiberseq_rate_cdr, colotb_fiberseq_rate_non_cdr) * 1.05,
             label=f"χ² p value = {colotb_fiberseq_res.pvalue:.2e}", size=9, color="black") +
    scale_fill_manual(values={"CDR": "#b89841", "Non-CDR": "#ac7daf"}) +
    scale_y_continuous(labels=scientific_format()) +
    labs(x="Region",
         y="Rate of unique unit-length α-satellite SVs (per bp)",
         title="COLO829 (Fiber-seq, ~87X)") +
    theme_minimal() +
    theme(
        figure_size=(4.5, 4),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
        )
)

ggsavefig_and_show(plot, "rate_of_unit_length_alpha_satellite_svs_colo829tb_cdr_vs_non_cdr")

# %%
# INFO: CDR vs. Non-CDR for COLO829BL and COLO829 together
# NOTE: Rate of unit-length α-satellite SVs within or outside CDR — COLO829BL & COLO829TB
sample_order = ["COLO829BL (Fiber-seq, ~170X)", "COLO829 (Fiber-seq, ~87X)"]

colo_rate_df = pd.DataFrame({
    "sample": pd.Categorical(
        ["COLO829BL (Fiber-seq, ~170X)", "COLO829BL (Fiber-seq, ~170X)",
         "COLO829 (Fiber-seq, ~87X)",    "COLO829 (Fiber-seq, ~87X)"],
        categories=sample_order, ordered=True,
    ),
    "region": ["CDR", "Non-CDR", "CDR", "Non-CDR"],
    "rate":   [colobl_fiberseq_rate_cdr, colobl_fiberseq_rate_non_cdr,
               colotb_fiberseq_rate_cdr, colotb_fiberseq_rate_non_cdr],
})

ymax = colo_rate_df["rate"].max() * 1.15

colo_pval_df = pd.DataFrame({
    "sample": pd.Categorical(sample_order, categories=sample_order, ordered=True),
    "x":      [1.5, 1.5],
    "y":      [ymax * 0.95, ymax * 0.95],
    "label":  [f"χ² p value = {colobl_fiberseq_res.pvalue:.2e}", f"χ² p value = {colotb_fiberseq_res.pvalue:.2e}"],
})

plot = (
    ggplot(colo_rate_df, aes(x="region", y="rate", fill="region")) +
    geom_col(width=0.7) +
    geom_text(aes(x="x", y="y", label="label"),
              data=colo_pval_df, inherit_aes=False, size=9, color="black") +
    facet_wrap("~sample", ncol=2) +
    scale_fill_manual(values={"CDR": "#b89841", "Non-CDR": "#ac7daf"}) +
    scale_y_continuous(labels=scientific_format(), limits=(0, ymax)) +
    labs(x="Region",
         y="Rate of unique unit-length α-satellite SVs (per bp)") +
    theme_minimal() +
    theme(
        figure_size=(8, 4),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "rate_of_unit_length_alpha_satellite_svs_colo829_bl_and_tb_cdr_vs_non_cdr")


# %%
# INFO: Histogram of Total "unique" SV lengths
plot = (ggplot(colotb_fiberseq_uid_sv_filtered_total_collapsed, aes(x='indel_length')) +
        geom_histogram(breaks=np.linspace(0, 20_000, 1_000), alpha=0.7, size=0.5, color='black') +
        scale_x_continuous(labels=comma_format()) +
        scale_y_log10(labels=comma_format()) +
        labs(title=f"SV Length Histogram (COLO829TB Fiber-seq)", 
             x='SV Length (bp)', 
             y='Count (log 10)') +
        theme_minimal() +
        theme(figure_size=(15, 5),
              text=element_text(family='Arial'),
              plot_title=element_text(size=15, color='black', hjust=0.5),
              axis_text_x=element_text(size=10, color='black'),
              axis_text_y=element_text(size=10, color='black'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, f"sv_length_histogram_colo829tb_fiberseq_uid_sv")

# %%
# INFO: Histogram of SV lengths (zoomed in up to 5,000 bp)
plot = (ggplot(colotb_fiberseq_uid_sv_filtered_total_collapsed, aes(x='indel_length')) +
        geom_histogram(breaks=np.linspace(0, 5_000, 1_000), alpha=0.7, size=0.5, color='black') +
        scale_x_continuous(labels=comma_format()) +
        scale_y_log10(labels=comma_format()) +
        geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*12, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*13, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*14, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*15, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*16, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*17, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*18, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*19, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*20, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*21, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*22, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*23, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*24, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*25, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*26, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*27, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*28, color="red", linetype="dashed", size=0.3) +
        geom_vline(xintercept=171*29, color="red", linetype="dashed", size=0.3) +
        annotate("text", x=171*1+10, y=100, label="171 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*2+10, y=100, label="171×2 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*3+10, y=100, label="171×3 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*4+10, y=100, label="171×4 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*5+10, y=100, label="171×5 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*6+10, y=100, label="171×6 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*7+10, y=100, label="171×7 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*8+10, y=100, label="171×8 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*9+10, y=100, label="171×9 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*10+10, y=100, label="171×10 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*11+10, y=100, label="171×11 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*12+10, y=100, label="171×12 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*13+10, y=100, label="171×13 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*14+10, y=100, label="171×14 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*15+10, y=100, label="171×15 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*16+10, y=100, label="171×16 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*17+10, y=100, label="171×17 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*18+10, y=100, label="171×18 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*19+10, y=100, label="171×19 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*20+10, y=100, label="171×20 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*21+10, y=100, label="171×21 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*22+10, y=100, label="171×22 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*23+10, y=100, label="171×23 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*24+10, y=100, label="171×24 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*25+10, y=100, label="171×25 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*26+10, y=100, label="171×26 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*27+10, y=100, label="171×27 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*28+10, y=100, label="171×28 bp", size=5, color="red", ha="center") +
        annotate("text", x=171*29+10, y=100, label="171×29 bp", size=5, color="red", ha="center") +
        labs(title=f"SV Length Histogram (COLO829 Fiber-seq), up to 5,000 bp only", 
             x='SV Length (bp)', 
             y='Count (log 10)') +
        theme_minimal() +
        theme(figure_size=(15, 5),
              text=element_text(family='Arial'),
              plot_title=element_text(size=15, color='black', hjust=0.5),
              axis_text_x=element_text(size=10, color='black'),
              axis_text_y=element_text(size=10, color='black'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, f"sv_length_histogram_colo829tb_fiberseq_uid_sv_zoomed_in_5000bp")

# %%
# INFO: Compute Autocorrelation 
_, colobl_fiberseq_uid_sv_filtered_total_collapsed_counts_table = get_sv_table_from_scanCSV(colobl_fiberseq_uid_sv_filtered_total_collapsed, min_sv_length=100)
_, colotb_fiberseq_uid_sv_filtered_total_collapsed_counts_table = get_sv_table_from_scanCSV(colotb_fiberseq_uid_sv_filtered_total_collapsed, min_sv_length=100)

# INFO: COLO829TB Fiber-seq
ac_colobl_fiberseq_uid_sv_max_lag2000 = autocorr_range(colobl_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=2000)
ac_colobl_fiberseq_uid_sv_max_lag3000 = autocorr_range(colobl_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=3000)
ac_colobl_fiberseq_uid_sv_max_lag15000 = autocorr_range(colobl_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=15000)


# INFO: COLO829TB Fiber-seq
ac_colotb_fiberseq_uid_sv_max_lag2000 = autocorr_range(colotb_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=2000)
ac_colotb_fiberseq_uid_sv_max_lag3000 = autocorr_range(colotb_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=3000)
ac_colotb_fiberseq_uid_sv_max_lag15000 = autocorr_range(colotb_fiberseq_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=15000)

# %%
# INFO: Plot Autocorrelation for COLO829BL Fiber-seq
# INFO: Plot Autocorrelation (lag up to 2,000)
plot = (
    ggplot(ac_colobl_fiberseq_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829BL)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_uid_sv_lag2000")

# INFO: Plot Autocorrelation (lag up to 15,000)
plot = (
    ggplot(ac_colobl_fiberseq_uid_sv_max_lag15000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829BL)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_uid_sv_lag15000")


# %%
# INFO: Plot Autocorrelation for COLO829TB Fiber-seq
# INFO: Plot Autocorrelation (lag up to 2,000)
plot = (
    ggplot(ac_colotb_fiberseq_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829tb_uid_sv_lag2000")

# INFO: Plot Autocorrelation (lag up to 15,000)
plot = (
    ggplot(ac_colotb_fiberseq_uid_sv_max_lag15000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829tb_uid_sv_lag15000")

# %%
# INFO: Overlay COLO829BL vs COLO829TB autocorrelation (lag up to 2,000)
ac_combined_lag2000 = pl.concat([
    ac_colobl_fiberseq_uid_sv_max_lag2000.with_columns(pl.lit("COLO829BL").alias("sample")),
    ac_colotb_fiberseq_uid_sv_max_lag2000.with_columns(pl.lit("COLO829TB").alias("sample")),
]).to_pandas()

plot = (
    ggplot(ac_combined_lag2000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values={"COLO829BL": "#196533", "COLO829TB": "#a97c50"}) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10,  y=0.3, label="171 bp",    size=8, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.3, label="171×2 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.3, label="171×3 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.3, label="171×4 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.3, label="171×5 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.3, label="171×6 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.3, label="171×7 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.3, label="171×8 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.3, label="171×9 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (COLO829BL vs COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_vs_colo829tb_uid_sv_lag2000")

# %%
# INFO: Overlay COLO829BL vs COLO829TB autocorrelation (lag up to 3,000)
ac_combined_lag3000 = pl.concat([
    ac_colobl_fiberseq_uid_sv_max_lag3000.with_columns(pl.lit("COLO829BL").alias("sample")),
    ac_colotb_fiberseq_uid_sv_max_lag3000.with_columns(pl.lit("COLO829TB").alias("sample")),
]).to_pandas()

plot = (
    ggplot(ac_combined_lag3000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values={"COLO829BL": "#ADEFD1", "COLO829TB": "#00203F"}) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*12, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*13, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*14, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*15, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*16, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*17, color="red", linetype="dashed", size=0.2) +
    annotate("text", x=171*1+10,  y=0.41, label="171 bp",    size=5, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.41, label="171×2 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.41, label="171×3 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.41, label="171×4 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.41, label="171×5 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.41, label="171×6 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.41, label="171×7 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.41, label="171×8 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.41, label="171×9 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.41, label="171×10 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.41, label="171×11 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*12+10, y=0.41, label="171×12 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*13+10, y=0.41, label="171×13 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*14+10, y=0.41, label="171×14 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*15+10, y=0.41, label="171×15 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*16+10, y=0.41, label="171×16 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*17+10, y=0.41, label="171×17 bp", size=5, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (COLO829BL vs COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_vs_colo829tb_uid_sv_lag3000")

# %%
# INFO: Kataegis events in COLO829
# NOTE: Loading up the somatic SNV sets for COLO829
colotb_snv = read_vcf("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.vcf.gz")

from pyfaidx import Fasta

def get_fasta_sequence(fasta_file, chrom: str, start: int, end: int) -> str:
    fasta = Fasta(fasta_file, rebuild=False)
    return fasta[chrom][start-1:end].seq


def reverse_complement(string):
    complement_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement_dict[s] for s in string[::-1])


def add_trinuc_context(df, fasta_file):
    """Attach trinucleotide context + APOBEC TCN flag to a VCF dataframe.

    Pyrimidine-strand convention: when REF is A/G, the trinuc, REF, and ALT
    are reverse-complemented so the middle base is always C or T.
    """
    fasta = Fasta(fasta_file, rebuild=False)
    ori_trinucs, trinucs, ref_pyr, alt_pyr = [],[], [], []
    for chrom, pos, ref, alt in zip(df["CHROM"], df["POS"], df["REF"], df["ALT"]):
        origin_tri = fasta[chrom][pos - 2:pos + 1].seq.upper()
        if ref in ("A", "G"):
            tri = reverse_complement(origin_tri)
            ref = reverse_complement(ref)
            alt = reverse_complement(alt)
        else:
            tri = origin_tri
        ori_trinucs.append(origin_tri)
        trinucs.append(tri)
        ref_pyr.append(ref)
        alt_pyr.append(alt)
    out = df.copy();
    out["ORIGIN_TRINUC"] = ori_trinucs
    out["TRINUC"] = trinucs
    out["REF_PYR"] = ref_pyr
    out["ALT_PYR"] = alt_pyr
    out["IS_TCN"] = [(r == "C") and t.startswith("T") and len(t) == 3
                    for r, t in zip(ref_pyr, trinucs)]
    return out


colotb_snv = add_trinuc_context(
    colotb_snv,
    "/mmfs1/gscratch/stergachislab/assemblies/DSA_COLO829BL_v3.0.0.fasta",
)

tpc_genome = colotb_snv[colotb_snv["IS_TCN"] == True].shape[0]
tpc_mutation_rate_genome = (tpc_genome / colo829bl_flagger_nucflag_100kb_del.length)

# %%
colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed = colotb_fiberseq_uid_sv_filtered_unit_collapsed.filter(
    pl.col("n_collapsed") >= 20
)

df = (
    colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed
    .to_pandas()[["#chrom", "start", "end"]]
    .copy()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)
df.loc[df["End"] - df["Start"] == 0, "Start"] -= 1

colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval = pr.PyRanges(df)

colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex1kb = colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval.extend_ranges(1_000).merge_overlaps()

colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex2kb = colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval.extend_ranges(2_000).merge_overlaps()

colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex10kb = colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval.extend_ranges(5_000).merge_overlaps()

colotb_snv_pr = pr.PyRanges(pd.DataFrame({
    "Chromosome": colotb_snv["CHROM"],
    "Start": colotb_snv["POS"] - 1,
    "End": colotb_snv["POS"],
    "ID": colotb_snv["IS_TCN"]
}))

colotb_snv_tcn_pr = colotb_snv_pr[colotb_snv_pr.ID == True]


tpc_unit_sv_up1kb_down1kb = colotb_snv_tcn_pr.intersect_overlaps(colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex1kb).shape[0]
tpc_unit_sv_up2kb_down2kb = colotb_snv_tcn_pr.intersect_overlaps(colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex2kb).shape[0]
tpc_unit_sv_up5kb_down5kb = colotb_snv_tcn_pr.intersect_overlaps(colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex10kb).shape[0]

tpc_mutation_rate_unitsv_up1kb_down1kb = (tpc_unit_sv_up1kb_down1kb / colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex1kb.length)
tpc_mutation_rate_unitsv_up2kb_down2kb = (tpc_unit_sv_up2kb_down2kb / colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex2kb.length)
tpc_mutation_rate_unitsv_up5kb_down5kb = (tpc_unit_sv_up5kb_down5kb / colotb_fiberseq_uid_sv_filtered_unit_collapsed_fixed_interval_ex10kb.length)

colo829bl_centromere_tumor = colo829bl_centromere.intersect_overlaps(colo829bl_flagger_nucflag_100kb_del)

tpc_centromere = colotb_snv_tcn_pr.intersect_overlaps(colo829bl_centromere_tumor).shape[0]
tpc_mutation_rate_centromere = (tpc_centromere / colo829bl_centromere_tumor.length)

print(f"TCN mutation rate in genome (per bp): {tpc_mutation_rate_genome}")
print(f"TCN mutation rate within centromere a-satellite regions (per bp): {tpc_mutation_rate_centromere}")
print(f"TCN mutation rate within 1kb of unit SVs (per bp): {tpc_mutation_rate_unitsv_up1kb_down1kb}")
print(f"TCN mutation rate within 2kb of unit SVs (per bp): {tpc_mutation_rate_unitsv_up2kb_down2kb}")
print(f"TCN mutation rate within 5kb of unit SVs (per bp): {tpc_mutation_rate_unitsv_up5kb_down5kb}")

# NOTE: Count the number of TpC mutations around translocations and calculate the rate and plot this
tpc_translocations_up1kb_down1kb = ((2+5)+(3+0)+(3+0))
tpc_translocations_up2kb_down2kb = (((2+10)+(4+0)+(6+0)))
tpc_translocations_up5kb_down5kb = (((2+15)+(4+0)+(6+0)))

tpc_mutation_rate_translocations_up1kb_down1kb = (tpc_translocations_up1kb_down1kb / 2_000)
tpc_mutation_rate_translocations_up2kb_down2kb = (tpc_translocations_up2kb_down2kb / 4_000)
tpc_mutation_rate_translocations_up5kb_down5kb = (tpc_translocations_up5kb_down5kb / 10_000)

tpc_translocations_up1kb_down1kb_only_a_satellite = ((3+0)+(3+0))
tpc_translocations_up2kb_down2kb_only_a_satellite = ((4+0)+(6+0))
tpc_translocations_up5kb_down5kb_only_a_satellite = ((4+0)+(6+0))

tpc_mutation_rate_translocations_up1kb_down1kb_only_a_satellite = (tpc_translocations_up1kb_down1kb_only_a_satellite / 2_000)
tpc_mutation_rate_translocations_up2kb_down2kb_only_a_satellite = (tpc_translocations_up2kb_down2kb_only_a_satellite / 4_000)
tpc_mutation_rate_translocations_up5kb_down5kb_only_a_satellite = (tpc_translocations_up5kb_down5kb_only_a_satellite / 10_000)

print(f"TCN mutation rate within 1kb of translocations (per bp): {tpc_mutation_rate_translocations_up1kb_down1kb}")
print(f"TCN mutation rate within 2kb of translocations (per bp): {tpc_mutation_rate_translocations_up2kb_down2kb}")
print(f"TCN mutation rate within 5kb of translocations (per bp): {tpc_mutation_rate_translocations_up5kb_down5kb}")

# %%
# NOTE: Barplot of TpC (TCN) mutation rates across genomic contexts in COLO829
region_order = [
    "Genome",
    "Centromere\nα-satellite",
    "Unit SV\n±1kb",
    "Unit SV\n±2kb",
    "Unit SV\n±5kb",
    "Translocation\n±1kb",
    "Translocation\n±2kb",
    "Translocation\n±5kb",
    "Translocation w a-sat\n±1kb",
    "Translocation w a-sat\n±2kb",
    "Translocation w a-sat\n±5kb",
]
category_order = ["Genome", "Centromere α-satellite", "Unit SV", "Translocation", "Translocation_only_a_satellite"]

tpc_rate_df = pd.DataFrame({
    "region": pd.Categorical(region_order, categories=region_order, ordered=True),
    "category": pd.Categorical(
        ["Genome", "Centromere α-satellite",
         "Unit SV", "Unit SV", "Unit SV",
         "Translocation", "Translocation", "Translocation",
         "Translocation_only_a_satellite", "Translocation_only_a_satellite", "Translocation_only_a_satellite"],
        categories=category_order, ordered=True,
    ),
    "rate": [
        tpc_mutation_rate_genome,
        tpc_mutation_rate_centromere,
        tpc_mutation_rate_unitsv_up1kb_down1kb,
        tpc_mutation_rate_unitsv_up2kb_down2kb,
        tpc_mutation_rate_unitsv_up5kb_down5kb,
        tpc_mutation_rate_translocations_up1kb_down1kb,
        tpc_mutation_rate_translocations_up2kb_down2kb,
        tpc_mutation_rate_translocations_up5kb_down5kb,
        tpc_mutation_rate_translocations_up1kb_down1kb_only_a_satellite,
        tpc_mutation_rate_translocations_up2kb_down2kb_only_a_satellite,
        tpc_mutation_rate_translocations_up5kb_down5kb_only_a_satellite
    ],
})

plot = (
    ggplot(tpc_rate_df, aes(x="region", y="rate", fill="category")) +
    geom_col(width=0.7) +
    scale_fill_manual(values={
        "Genome": "#8c8c8c",
        "Centromere α-satellite": "#b89841",
        "Unit SV": "#ac7daf",
        "Translocation": "#c0504d",
        "Translocation_only_a_satellite": "#800000",
    }) +
    scale_y_log10(labels=scientific_format()) +
    labs(x="Region",
         y="TpC (TCN) mutation rate (per bp)",
         title="COLO829 TpC mutation rate by genomic context") +
    theme_minimal() +
    theme(
        figure_size=(9, 4.5),
        text=element_text(family='Arial'),
        plot_title=element_text(size=13, color='black', hjust=0.5),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "tpc_mutation_rate_by_genomic_context_colo829")


# %%
# INFO: Benchmarking donor tissues
centroindel_benchmark_dir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/SV_Centromere"

# INFO: ST001 Liver
st001_liver_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST001_Liver/ST001_Liver_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st001_liver_uid_sv, st001_liver_uid_sv_counts_table = get_sv_table_from_scanCSV(st001_liver_uid)

# INFO: ST001 Lung
st001_lung_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST001_Lung/ST001_Lung_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st001_lung_uid_sv, st001_lung_uid_sv_counts_table = get_sv_table_from_scanCSV(st001_lung_uid)

# INFO: ST002 Colon
st002_colon_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST002_Colon/ST002_Colon_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st002_colon_uid_sv, st002_colon_uid_sv_counts_table = get_sv_table_from_scanCSV(st002_colon_uid)

# INFO: ST002 Lung
st002_lung_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST002_Lung/ST002_Lung_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st002_lung_uid_sv, st002_lung_uid_sv_counts_table = get_sv_table_from_scanCSV(st002_lung_uid)

# INFO: ST003 Brain
st003_brain_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST003_Brain/ST003_Brain_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st003_brain_uid_sv, st003_brain_uid_sv_counts_table = get_sv_table_from_scanCSV(st003_brain_uid)

# INFO: ST004 Brain
st004_brain_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST004_Brain/ST004_Brain_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st004_brain_uid_sv, st004_brain_uid_sv_counts_table = get_sv_table_from_scanCSV(st004_brain_uid)

# %%
# INFO: Several annotations for benchmarking tissue DSAs
# INFO: ST001 
st001_contig_chrom_assign = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/DSA_ST001_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv", separator="\t", has_header=True)

st001_contig_length = pl.read_csv("/mmfs1/gscratch/stergachislab/assemblies/DSA_smaht/DSA_ST001/hifiasm/0.25.0/DSA_ST001.v1.0.0.fasta.fai", separator="\t", has_header=False, columns=[0, 1], new_columns=["ID", "Length"])

st001_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST001/centromere/DSA_ST001_1_2_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st001_flagger = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Flagger_NucFreq_NucFlag/DSA_ST001/flagger/ST001_DSA_v1.hifi_winnowmap_flagger_040.flagger_final_misassembly.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st001_liver_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST001/centromere/CDR/FromPacBio/ST001_Liver_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st001_liver_cdr = st001_liver_cdr.extend_ranges(50_000).merge_overlaps()

st001_lung_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST001/centromere/CDR/FromPacBio/ST001_Lung_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st001_lung_cdr = st001_lung_cdr.extend_ranges(50_000).merge_overlaps()

# INFO: ST002
st002_contig_chrom_assign = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/DSA_ST002_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv", separator="\t", has_header=True)

st002_contig_length = pl.read_csv("/mmfs1/gscratch/stergachislab/assemblies/DSA_smaht/DSA_ST002/hifiasm/0.25.0/DSA_ST002.v1.0.0.fasta.fai", separator="\t", has_header=False, columns=[0, 1], new_columns=["ID", "Length"])

st002_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST002/centromere/DSA_ST002_1_2_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st002_flagger = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Flagger_NucFreq_NucFlag/DSA_ST002/flagger/ST002_DSA_v1.hifi_winnowmap_flagger_040.flagger_final_misassembly.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st002_colon_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST002/centromere/CDR/FromPacBio/ST002_Colon_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st002_colon_cdr = st002_colon_cdr.extend_ranges(50_000).merge_overlaps()

st002_lung_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST002/centromere/CDR/FromPacBio/ST002_Lung_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st002_lung_cdr = st002_lung_cdr.extend_ranges(50_000).merge_overlaps()

# INFO: ST003
st003_contig_chrom_assign = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/DSA_ST003_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv", separator="\t", has_header=True)

st003_contig_length = pl.read_csv("/mmfs1/gscratch/stergachislab/assemblies/DSA_smaht/DSA_ST003/hifiasm/0.25.0/DSA_ST003.v1.0.0.fasta.fai", separator="\t", has_header=False, columns=[0, 1], new_columns=["ID", "Length"])

st003_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST003/centromere/DSA_ST003_1_2_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st003_flagger = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Flagger_NucFreq_NucFlag/DSA_ST003/flagger/ST003_DSA_v1.hifi_winnowmap_flagger_040.flagger_final_misassembly.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st003_brain_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST003/centromere/CDR/FromPacBio/ST003_Brain_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st003_brain_cdr = st003_brain_cdr.extend_ranges(50_000).merge_overlaps()


# INFO: ST004
st004_contig_chrom_assign = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/DSA_ST004_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv", separator="\t", has_header=True)

st004_contig_length = pl.read_csv("/mmfs1/gscratch/stergachislab/assemblies/DSA_smaht/DSA_ST004/hifiasm/0.25.0/DSA_ST004.v1.0.0.fasta.fai", separator="\t", has_header=False, columns=[0, 1], new_columns=["ID", "Length"])

st004_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST004/centromere/DSA_ST004_1_2_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st004_flagger = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Flagger_NucFreq_NucFlag/DSA_ST004/flagger/ST004_DSA_v1.hifi_winnowmap_flagger_040.flagger_final_misassembly.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

st004_brain_cdr = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST004/centromere/CDR/FromPacBio/ST004_Brain_merged_resetmapq.CDR.live.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
st004_brain_cdr = st004_brain_cdr.extend_ranges(50_000).merge_overlaps()

# INFO: LB-LA2
"""
lbla2_contig_chrom_assign = pl.read_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/DSA_ST001_1_2_contig_to_t2t-chm13_primary_chromosome_assignment.tsv", separator="\t", has_header=True)

lbla2_contig_length = pl.read_csv("/mmfs1/gscratch/stergachislab/assemblies/DSA_smaht/DSA_ST001/hifiasm/0.25.0/DSA_ST001.v1.0.0.fasta.fai", separator="\t", has_header=False, columns=[0, 1], new_columns=["ID", "Length"])

lbla2_centromere = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Analysis/ST001/centromere/DSA_ST001_1_2_CHM13-centromere.stats.DSA.start2end.ALR_Alpha.100kb.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))

lbla2_flagger = pr.PyRanges(pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Benchmark_Tissues/Flagger_NucFreq_NucFlag/DSA_ST001/flagger/ST001_DSA_v1.hifi_winnowmap_flagger_040.flagger_final_misassembly.bed.gz", header=None, sep="\t").iloc[:, :3].set_axis(["Chromosome", "Start", "End"], axis=1))
"""
# %%
# INFO: ST001 Lung
st001_lung_uid_sv_pr = pr.PyRanges(
    st001_lung_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st001_lung_uid_sv_pr_st001_flagger_overlap_idx = st001_lung_uid_sv_pr.overlap(st001_flagger)["_idx"].values

st001_lung_uid_sv_filtered = (st001_lung_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st001_lung_uid_sv_pr_st001_flagger_overlap_idx))
    .drop("_idx")) 

st001_lung_uid_sv_filtered_total = st001_lung_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st001_lung_uid_sv_filtered_unit = st001_lung_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st001_lung_uid_sv_filtered_nonunit = st001_lung_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st001_lung_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st001_lung_uid_sv_filtered_unit)
st001_lung_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st001_lung_uid_sv_filtered_nonunit)

st001_lung_uid_sv_filtered_unit_total_collapsed = pl.concat([st001_lung_uid_sv_filtered_unit_collapsed, st001_lung_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st001_lung_uid_sv_counts_table = st001_lung_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st001_lung_uid_sv_unit_sv_cdr = st001_lung_uid_sv_counts_table.filter(st001_lung_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st001_lung_uid_sv_unit_sv_non_cdr = st001_lung_uid_sv_counts_table.filter(st001_lung_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST001 Lung
st001_lung_fiberseq_contingency_table = [
    [st001_lung_uid_sv_unit_sv_cdr, st001_lung_uid_sv_unit_sv_non_cdr],
    [st001_lung_cdr.subtract_overlaps(st001_flagger).length, st001_centromere.subtract_overlaps(st001_flagger).length - st001_lung_cdr.subtract_overlaps(st001_flagger).length]
    ]

st001_lung_fiberseq_res = chi2_contingency(st001_lung_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st001_lung_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST001 Lung:{st001_lung_uid_sv_unit_sv_cdr / st001_lung_cdr.subtract_overlaps(st001_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST001 Lung:{st001_lung_uid_sv_unit_sv_non_cdr / (st001_centromere.subtract_overlaps(st001_flagger).length - st001_lung_cdr.subtract_overlaps(st001_flagger).length)}")

# NOTE: Good candidate : h2tg000033l:93622262-93622262

# %%
# INFO: ST001 Liver
st001_liver_uid_sv_pr = pr.PyRanges(
    st001_liver_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st001_liver_uid_sv_pr_st001_flagger_overlap_idx = st001_liver_uid_sv_pr.overlap(st001_flagger)["_idx"].values

st001_liver_uid_sv_filtered = (st001_liver_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st001_liver_uid_sv_pr_st001_flagger_overlap_idx))
    .drop("_idx"))

st001_liver_uid_sv_filtered_total = st001_liver_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st001_liver_uid_sv_filtered_unit = st001_liver_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st001_liver_uid_sv_filtered_nonunit = st001_liver_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st001_liver_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st001_liver_uid_sv_filtered_unit)
st001_liver_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st001_liver_uid_sv_filtered_nonunit)

st001_liver_uid_sv_filtered_unit_total_collapsed = pl.concat([st001_liver_uid_sv_filtered_unit_collapsed, st001_liver_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st001_liver_uid_sv_counts_table = st001_liver_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st001_liver_uid_sv_unit_sv_cdr = st001_liver_uid_sv_counts_table.filter(st001_liver_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st001_liver_uid_sv_unit_sv_non_cdr = st001_liver_uid_sv_counts_table.filter(st001_liver_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST001 Liver
st001_liver_fiberseq_contingency_table = [
    [st001_liver_uid_sv_unit_sv_cdr, st001_liver_uid_sv_unit_sv_non_cdr],
    [st001_liver_cdr.subtract_overlaps(st001_flagger).length, st001_centromere.subtract_overlaps(st001_flagger).length - st001_liver_cdr.subtract_overlaps(st001_flagger).length]
    ]

st001_liver_fiberseq_res = chi2_contingency(st001_liver_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st001_liver_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST001 Liver:{st001_liver_uid_sv_unit_sv_cdr / st001_liver_cdr.subtract_overlaps(st001_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST001 Liver:{st001_liver_uid_sv_unit_sv_non_cdr / (st001_centromere.subtract_overlaps(st001_flagger).length - st001_liver_cdr.subtract_overlaps(st001_flagger).length)}")


# %%
# %%
# INFO: ST002 Lung
st002_lung_uid_sv_pr = pr.PyRanges(
    st002_lung_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st002_lung_uid_sv_pr_st002_flagger_overlap_idx = st002_lung_uid_sv_pr.overlap(st002_flagger)["_idx"].values

st002_lung_uid_sv_filtered = (st002_lung_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st002_lung_uid_sv_pr_st002_flagger_overlap_idx))
    .drop("_idx"))

st002_lung_uid_sv_filtered_total = st002_lung_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st002_lung_uid_sv_filtered_unit = st002_lung_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st002_lung_uid_sv_filtered_nonunit = st002_lung_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st002_lung_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st002_lung_uid_sv_filtered_unit)
st002_lung_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st002_lung_uid_sv_filtered_nonunit)

st002_lung_uid_sv_filtered_unit_total_collapsed = pl.concat([st002_lung_uid_sv_filtered_unit_collapsed, st002_lung_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st002_lung_uid_sv_counts_table = st002_lung_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st002_lung_uid_sv_unit_sv_cdr = st002_lung_uid_sv_counts_table.filter(st002_lung_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st002_lung_uid_sv_unit_sv_non_cdr = st002_lung_uid_sv_counts_table.filter(st002_lung_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST002 Lung
st002_lung_fiberseq_contingency_table = [
    [st002_lung_uid_sv_unit_sv_cdr, st002_lung_uid_sv_unit_sv_non_cdr],
    [st002_lung_cdr.subtract_overlaps(st002_flagger).length, st002_centromere.subtract_overlaps(st002_flagger).length - st002_lung_cdr.subtract_overlaps(st002_flagger).length]
    ]

st002_lung_fiberseq_res = chi2_contingency(st002_lung_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st002_lung_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST002 Lung:{st002_lung_uid_sv_unit_sv_cdr / st002_lung_cdr.subtract_overlaps(st002_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST002 Lung:{st002_lung_uid_sv_unit_sv_non_cdr / (st002_centromere.subtract_overlaps(st002_flagger).length - st002_lung_cdr.subtract_overlaps(st002_flagger).length)}")


# %%
# INFO: ST002 Colon
st002_colon_uid_sv_pr = pr.PyRanges(
    st002_colon_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st002_colon_uid_sv_pr_st002_flagger_overlap_idx = st002_colon_uid_sv_pr.overlap(st002_flagger)["_idx"].values

st002_colon_uid_sv_filtered = (st002_colon_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st002_colon_uid_sv_pr_st002_flagger_overlap_idx))
    .drop("_idx"))

st002_colon_uid_sv_filtered_total = st002_colon_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st002_colon_uid_sv_filtered_unit = st002_colon_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st002_colon_uid_sv_filtered_nonunit = st002_colon_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st002_colon_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st002_colon_uid_sv_filtered_unit)
st002_colon_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st002_colon_uid_sv_filtered_nonunit)

st002_colon_uid_sv_filtered_unit_total_collapsed = pl.concat([st002_colon_uid_sv_filtered_unit_collapsed, st002_colon_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st002_colon_uid_sv_counts_table = st002_colon_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st002_colon_uid_sv_unit_sv_cdr = st002_colon_uid_sv_counts_table.filter(st002_colon_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st002_colon_uid_sv_unit_sv_non_cdr = st002_colon_uid_sv_counts_table.filter(st002_colon_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST002 Colon
st002_colon_fiberseq_contingency_table = [
    [st002_colon_uid_sv_unit_sv_cdr, st002_colon_uid_sv_unit_sv_non_cdr],
    [st002_colon_cdr.subtract_overlaps(st002_flagger).length, st002_centromere.subtract_overlaps(st002_flagger).length - st002_colon_cdr.subtract_overlaps(st002_flagger).length]
    ]

st002_colon_fiberseq_res = chi2_contingency(st002_colon_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st002_colon_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST002 Colon:{st002_colon_uid_sv_unit_sv_cdr / st002_colon_cdr.subtract_overlaps(st002_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST002 Colon:{st002_colon_uid_sv_unit_sv_non_cdr / (st002_centromere.subtract_overlaps(st002_flagger).length - st002_colon_cdr.subtract_overlaps(st002_flagger).length)}")

# %%
# INFO: ST003 Brain
st003_brain_uid_sv_pr = pr.PyRanges(
    st003_brain_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st003_brain_uid_sv_pr_st003_flagger_overlap_idx = st003_brain_uid_sv_pr.overlap(st003_flagger)["_idx"].values

st003_brain_uid_sv_filtered = (st003_brain_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st003_brain_uid_sv_pr_st003_flagger_overlap_idx))
    .drop("_idx"))

st003_brain_uid_sv_filtered_total = st003_brain_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st003_brain_uid_sv_filtered_unit = st003_brain_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st003_brain_uid_sv_filtered_nonunit = st003_brain_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st003_brain_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st003_brain_uid_sv_filtered_unit)
st003_brain_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st003_brain_uid_sv_filtered_nonunit)

st003_brain_uid_sv_filtered_unit_total_collapsed = pl.concat([st003_brain_uid_sv_filtered_unit_collapsed, st003_brain_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st003_brain_uid_sv_counts_table = st003_brain_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st003_brain_uid_sv_unit_sv_cdr = st003_brain_uid_sv_counts_table.filter(st003_brain_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st003_brain_uid_sv_unit_sv_non_cdr = st003_brain_uid_sv_counts_table.filter(st003_brain_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST003 Brain
st003_brain_fiberseq_contingency_table = [
    [st003_brain_uid_sv_unit_sv_cdr, st003_brain_uid_sv_unit_sv_non_cdr],
    [st003_brain_cdr.subtract_overlaps(st003_flagger).length, st003_centromere.subtract_overlaps(st003_flagger).length - st003_brain_cdr.subtract_overlaps(st003_flagger).length]
    ]

st003_brain_fiberseq_res = chi2_contingency(st003_brain_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st003_brain_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST003 Brain:{st003_brain_uid_sv_unit_sv_cdr / st003_brain_cdr.subtract_overlaps(st003_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST003 Brain:{st003_brain_uid_sv_unit_sv_non_cdr / (st003_centromere.subtract_overlaps(st003_flagger).length - st003_brain_cdr.subtract_overlaps(st003_flagger).length)}")

# %%
# INFO: ST004 Brain
st004_brain_uid_sv_pr = pr.PyRanges(
    st004_brain_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st004_brain_uid_sv_pr_st004_flagger_overlap_idx = st004_brain_uid_sv_pr.overlap(st004_flagger)["_idx"].values

st004_brain_uid_sv_filtered = (st004_brain_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st004_brain_uid_sv_pr_st004_flagger_overlap_idx))
    .drop("_idx"))

st004_brain_uid_sv_filtered_total = st004_brain_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.998) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st004_brain_uid_sv_filtered_unit = st004_brain_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st004_brain_uid_sv_filtered_nonunit = st004_brain_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st004_brain_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st004_brain_uid_sv_filtered_unit)
st004_brain_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st004_brain_uid_sv_filtered_nonunit)

st004_brain_uid_sv_filtered_unit_total_collapsed = pl.concat([st004_brain_uid_sv_filtered_unit_collapsed, st004_brain_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

st004_brain_uid_sv_counts_table = st004_brain_uid_sv_filtered_unit_collapsed["overlap_class"].value_counts()
st004_brain_uid_sv_unit_sv_cdr = st004_brain_uid_sv_counts_table.filter(st004_brain_uid_sv_counts_table["overlap_class"] == "target")["count"][0]
st004_brain_uid_sv_unit_sv_non_cdr = st004_brain_uid_sv_counts_table.filter(st004_brain_uid_sv_counts_table["overlap_class"] == "non_target")["count"][0]

# INFO: CDR vs. Non-CDR for ST004 Brain
st004_brain_fiberseq_contingency_table = [
    [st004_brain_uid_sv_unit_sv_cdr, st004_brain_uid_sv_unit_sv_non_cdr],
    [st004_brain_cdr.subtract_overlaps(st004_flagger).length, st004_centromere.subtract_overlaps(st004_flagger).length - st004_brain_cdr.subtract_overlaps(st004_flagger).length]
    ]

st004_brain_fiberseq_res = chi2_contingency(st004_brain_fiberseq_contingency_table)
print(f"Chi-squared p-value: {st004_brain_fiberseq_res.pvalue}")
print(f"Rate of unit-length α-satellite SVs within CDR of ST004 Brain:{st004_brain_uid_sv_unit_sv_cdr / st004_brain_cdr.subtract_overlaps(st004_flagger).length}")
print(f"Rate of unit-length α-satellite SVs outside CDR of ST004 Brain:{st004_brain_uid_sv_unit_sv_non_cdr / (st004_centromere.subtract_overlaps(st004_flagger).length - st004_brain_cdr.subtract_overlaps(st004_flagger).length)}")

# %%
_, st001_lung_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st001_lung_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st001_liver_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st001_liver_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st002_lung_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st002_lung_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st002_colon_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st002_colon_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st003_brain_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st003_brain_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st004_brain_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st004_brain_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)


# %%
# INFO: Compute Autocorrelation
ac_st001_liver_uid_sv_max_lag2000 = autocorr_range(st001_liver_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st001_lung_uid_sv_max_lag2000 = autocorr_range(st001_lung_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st002_colon_uid_sv_max_lag2000 = autocorr_range(st002_colon_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st002_lung_uid_sv_max_lag2000 = autocorr_range(st002_lung_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st003_brain_uid_sv_max_lag2000 = autocorr_range(st003_brain_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st004_brain_uid_sv_max_lag2000 = autocorr_range(st004_brain_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)

# %%
# INFO: Plot Autocorrelation for ST001 Liver (lag up to 2,000)
plot = (
    ggplot(ac_st001_liver_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST001 Liver)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st001_liver_uid_sv_lag2000")

# INFO: Plot Autocorrelation for ST001 Lung (lag up to 2,000)
plot = (
    ggplot(ac_st001_lung_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST001 Lung)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st001_lung_uid_sv_lag2000")

# INFO: Plot Autocorrelation for ST002 Colon (lag up to 2,000)
plot = (
    ggplot(ac_st002_colon_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST002 Colon)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st002_colon_uid_sv_lag2000")

# INFO: Plot Autocorrelation for ST002 Lung (lag up to 2,000)
plot = (
    ggplot(ac_st002_lung_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST002 Lung)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st002_lung_uid_sv_lag2000")

# INFO: Plot Autocorrelation for ST003 Brain (lag up to 2,000)
plot = (
    ggplot(ac_st003_brain_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST003 Brain)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st003_brain_uid_sv_lag2000")

# INFO: Plot Autocorrelation for ST004 Brain (lag up to 2,000)
plot = (
    ggplot(ac_st004_brain_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (ST004 Brain)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_st004_brain_uid_sv_lag2000")

# %%
# INFO: Overlay all 6 benchmark tissues autocorrelation (lag up to 2,000)
benchmark_sample_order = [
    "ST001 Liver", "ST001 Lung",
    "ST002 Colon", "ST002 Lung",
    "ST003 Brain", "ST004 Brain",
]

ac_benchmark_combined_lag2000 = pl.concat([
    ac_st001_liver_uid_sv_max_lag2000.with_columns(pl.lit("ST001 Liver").alias("sample")),
    ac_st001_lung_uid_sv_max_lag2000.with_columns(pl.lit("ST001 Lung").alias("sample")),
    ac_st002_colon_uid_sv_max_lag2000.with_columns(pl.lit("ST002 Colon").alias("sample")),
    ac_st002_lung_uid_sv_max_lag2000.with_columns(pl.lit("ST002 Lung").alias("sample")),
    ac_st003_brain_uid_sv_max_lag2000.with_columns(pl.lit("ST003 Brain").alias("sample")),
    ac_st004_brain_uid_sv_max_lag2000.with_columns(pl.lit("ST004 Brain").alias("sample")),
]).to_pandas()

ac_benchmark_combined_lag2000["sample"] = pd.Categorical(
    ac_benchmark_combined_lag2000["sample"],
    categories=benchmark_sample_order, ordered=True,
)

benchmark_color_map = {
    "ST001 Liver": "#01befe",
    "ST001 Lung":  "#ffdd00",
    "ST002 Colon": "#ff7d00",
    "ST002 Lung":  "#ff006d",
    "ST003 Brain": "#adff02",
    "ST004 Brain": "#8f00ff",
}

plot = (
    ggplot(ac_benchmark_combined_lag2000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values=benchmark_color_map) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.3) +
    annotate("text", x=171*1+10,  y=0.3, label="171 bp",    size=8, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.3, label="171×2 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.3, label="171×3 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.3, label="171×4 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.3, label="171×5 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.3, label="171×6 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.3, label="171×7 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.3, label="171×8 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.3, label="171×9 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (Benchmark tissues)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_benchmark_tissues_combined_uid_sv_lag2000")


# %%
# INFO: Obtaining SV sets for COLO829BL and COLO829 
# INFO: COLO829BL UL-ONT
colobl_ulont_uid_sv, colobl_ulont_uid_sv_counts_table = get_sv_table_from_scanCSV(colobl_ulont_uid)

colobl_ulont_uid_sv_pr = pr.PyRanges(
    colobl_ulont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") - 1) # NOTE: For INS, start == end, so need to fix
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

colobl_ulont_uid_sv_pr_colo829bl_flagger_nucflag_overlap_idx = colobl_ulont_uid_sv_pr.overlap(colo829bl_flagger_nucflag)["_idx"].values

colobl_ulont_uid_sv_filtered = (colobl_ulont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(colobl_ulont_uid_sv_pr_colo829bl_flagger_nucflag_overlap_idx))
    .drop("_idx"))

colobl_ulont_uid_sv_filtered_total = colobl_ulont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.97) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

# INFO: COLO829TB UL-ONT
colotb_ulont_uid_sv, colotb_ulont_uid_sv_counts_table = get_sv_table_from_scanCSV(colotb_ulont_uid)

colotb_ulont_uid_sv_pr = pr.PyRanges(
    colotb_ulont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1) # NOTE: For INS, start == end, so need to fix
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

colotb_ulont_uid_sv_pr_colo829bl_flagger_nucflag_100kb_del_overlap_idx = colotb_ulont_uid_sv_pr.overlap(colo829bl_flagger_nucflag_100kb_del)["_idx"].values

colotb_ulont_uid_sv_filtered = (colotb_ulont_uid_sv
    .with_row_index("_idx")
    .filter(pl.col("_idx").is_in(colotb_ulont_uid_sv_pr_colo829bl_flagger_nucflag_100kb_del_overlap_idx))
    .drop("_idx")) # NOTE: Here, instead of filtering out SNVs that overlap, we are keeping only those that overlap with the 100 kb flagged regions.

colotb_ulont_uid_sv_filtered_total = colotb_ulont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.97) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

# INFO: Obtaining the overallped SVs (Added)
colobl_colotb_ulont_overlapping_indel_id = set(colobl_ulont_uid_sv_filtered_total["indel_id"].to_list()) & set(colotb_ulont_uid_sv_filtered_total["indel_id"].to_list())

# INFO: Removing the overlapped SVs (Added)
# INFO: COLO829BL UL-ONT
colobl_ulont_uid_sv_filtered_total = colobl_ulont_uid_sv_filtered_total.filter(
    ~pl.col("indel_id").is_in(colobl_colotb_ulont_overlapping_indel_id)
)

colobl_ulont_uid_sv_filtered_unit = colobl_ulont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
    )

colobl_ulont_uid_sv_filtered_nonunit = colobl_ulont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
    )

colobl_ulont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(colobl_ulont_uid_sv_filtered_unit)
colobl_ulont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(colobl_ulont_uid_sv_filtered_nonunit)

colobl_ulont_uid_sv_filtered_total_collapsed = pl.concat([colobl_ulont_uid_sv_filtered_unit_collapsed, colobl_ulont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: Removing the overlapped SVs (Added)
# INFO: COLO829TB UL-ONT
colotb_ulont_uid_sv_filtered_total = colotb_ulont_uid_sv_filtered_total.filter(
    ~pl.col("indel_id").is_in(colobl_colotb_ulont_overlapping_indel_id)
)

colotb_ulont_uid_sv_filtered_unit = colotb_ulont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
    )

colotb_ulont_uid_sv_filtered_nonunit = colotb_ulont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
    )

colotb_ulont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(colotb_ulont_uid_sv_filtered_unit)
colotb_ulont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(colotb_ulont_uid_sv_filtered_nonunit)

colotb_ulont_uid_sv_filtered_total_collapsed = pl.concat([colotb_ulont_uid_sv_filtered_unit_collapsed, colotb_ulont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])


# %%
# INFO: Compute Autocorrelation 
_, colobl_ulont_uid_sv_filtered_total_collapsed_counts_table = get_sv_table_from_scanCSV(colobl_ulont_uid_sv_filtered_total_collapsed, min_sv_length=100)
_, colotb_ulont_uid_sv_filtered_total_collapsed_counts_table = get_sv_table_from_scanCSV(colotb_ulont_uid_sv_filtered_total_collapsed, min_sv_length=100)

# INFO: COLO829BL UL-ONT
ac_colobl_ulont_uid_sv_max_lag2000 = autocorr_range(colobl_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=2000)
ac_colobl_ulont_uid_sv_max_lag3000 = autocorr_range(colobl_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=3000)
ac_colobl_ulont_uid_sv_max_lag5000 = autocorr_range(colobl_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=5000)
ac_colobl_ulont_uid_sv_max_lag15000 = autocorr_range(colobl_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=15000)


# INFO: COLO829TB UL-ONT
ac_colotb_ulont_uid_sv_max_lag2000 = autocorr_range(colotb_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=2000)
ac_colotb_ulont_uid_sv_max_lag3000 = autocorr_range(colotb_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=3000)
ac_colotb_ulont_uid_sv_max_lag5000 = autocorr_range(colotb_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=5000)
ac_colotb_ulont_uid_sv_max_lag15000 = autocorr_range(colotb_ulont_uid_sv_filtered_total_collapsed_counts_table["count"], max_lag=15000)

# INFO: Plot Autocorrelation for COLO829BL UL-ONT
# INFO: Plot Autocorrelation (lag up to 2,000)
plot = (
    ggplot(ac_colobl_ulont_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829BL)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_ulont_uid_sv_lag2000")

# INFO: Plot Autocorrelation (lag up to 15,000)
plot = (
    ggplot(ac_colobl_ulont_uid_sv_max_lag15000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829BL)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_ulont_uid_sv_lag15000")

# INFO: Plot Autocorrelation for COLO829TB Fiber-seq
# INFO: Plot Autocorrelation (lag up to 2,000)
plot = (
    ggplot(ac_colotb_ulont_uid_sv_max_lag2000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10, y=0.3, label="171 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*2+10, y=0.3, label="171×2 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*3+10, y=0.3, label="171×3 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*4+10, y=0.3, label="171×4 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*5+10, y=0.3, label="171×5 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*6+10, y=0.3, label="171×6 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*7+10, y=0.3, label="171×7 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*8+10, y=0.3, label="171×8 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*9+10, y=0.3, label="171×9 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829tb_ulont_uid_sv_lag2000")

# INFO: Plot Autocorrelation (lag up to 15,000)
plot = (
    ggplot(ac_colotb_ulont_uid_sv_max_lag15000, aes(x="lag", y="autocorr")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    labs(x="Lag between SV length vectors", y="Autocorrelation between SV events", title="Autocorrelation of SV lengths within Centromeres (COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray')
        )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829tb_ulont_uid_sv_lag15000")

# INFO: Overlay COLO829BL vs COLO829TB autocorrelation (lag up to 2,000)
ac_ulont_combined_lag2000 = pl.concat([
    ac_colobl_ulont_uid_sv_max_lag2000.with_columns(pl.lit("COLO829BL").alias("sample")),
    ac_colotb_ulont_uid_sv_max_lag2000.with_columns(pl.lit("COLO829TB").alias("sample")),
]).to_pandas()

plot = (
    ggplot(ac_ulont_combined_lag2000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values={"COLO829BL": "#196533", "COLO829TB": "#a97c50"}) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed") +
    geom_vline(xintercept=171*3, color="red", linetype="dashed") +
    geom_vline(xintercept=171*4, color="red", linetype="dashed") +
    geom_vline(xintercept=171*5, color="red", linetype="dashed") +
    geom_vline(xintercept=171*6, color="red", linetype="dashed") +
    geom_vline(xintercept=171*7, color="red", linetype="dashed") +
    geom_vline(xintercept=171*8, color="red", linetype="dashed") +
    geom_vline(xintercept=171*9, color="red", linetype="dashed") +
    geom_vline(xintercept=171*10, color="red", linetype="dashed") +
    geom_vline(xintercept=171*11, color="red", linetype="dashed") +
    annotate("text", x=171*1+10,  y=0.3, label="171 bp",    size=8, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.3, label="171×2 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.3, label="171×3 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.3, label="171×4 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.3, label="171×5 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.3, label="171×6 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.3, label="171×7 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.3, label="171×8 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.3, label="171×9 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (COLO829BL vs COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_vs_colo829tb_ulont_uid_sv_lag2000") 

# INFO: Overlay COLO829BL vs COLO829TB autocorrelation (lag up to 3,000)
ac_ulont_combined_lag3000 = pl.concat([
    ac_colobl_ulont_uid_sv_max_lag3000.with_columns(pl.lit("COLO829BL").alias("sample")),
    ac_colotb_ulont_uid_sv_max_lag3000.with_columns(pl.lit("COLO829TB").alias("sample")),
]).to_pandas()

plot = (
    ggplot(ac_ulont_combined_lag3000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values={"COLO829BL": "#ADEFD1", "COLO829TB": "#00203F"}) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*12, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*13, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*14, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*15, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*16, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*17, color="red", linetype="dashed", size=0.2) +
    annotate("text", x=171*1+10,  y=0.41, label="171 bp",    size=5, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.41, label="171×2 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.41, label="171×3 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.41, label="171×4 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.41, label="171×5 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.41, label="171×6 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.41, label="171×7 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.41, label="171×8 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.41, label="171×9 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.41, label="171×10 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.41, label="171×11 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*12+10, y=0.41, label="171×12 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*13+10, y=0.41, label="171×13 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*14+10, y=0.41, label="171×14 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*15+10, y=0.41, label="171×15 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*16+10, y=0.41, label="171×16 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*17+10, y=0.41, label="171×17 bp", size=5, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (COLO829BL vs COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_vs_colo829tb_ulont_uid_sv_lag3000")

# INFO: Overlay COLO829BL vs COLO829TB autocorrelation (lag up to 5,000)
ac_ulont_combined_lag5000 = pl.concat([
    ac_colobl_ulont_uid_sv_max_lag5000.with_columns(pl.lit("COLO829BL").alias("sample")),
    ac_colotb_ulont_uid_sv_max_lag5000.with_columns(pl.lit("COLO829TB").alias("sample")),
]).to_pandas()

plot = (
    ggplot(ac_ulont_combined_lag5000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values={"COLO829BL": "#ADEFD1", "COLO829TB": "#00203F"}) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*12, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*13, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*14, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*15, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*16, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*17, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*18, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*19, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*20, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*21, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*22, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*23, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*24, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*25, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*26, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*27, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*28, color="red", linetype="dashed", size=0.2) +
    geom_vline(xintercept=171*29, color="red", linetype="dashed", size=0.2) +
    annotate("text", x=171*1+10,  y=0.41, label="171 bp",    size=5, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.41, label="171×2 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.41, label="171×3 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.41, label="171×4 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.41, label="171×5 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.41, label="171×6 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.41, label="171×7 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.41, label="171×8 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.41, label="171×9 bp",  size=5, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.41, label="171×10 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.41, label="171×11 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*12+10, y=0.41, label="171×12 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*13+10, y=0.41, label="171×13 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*14+10, y=0.41, label="171×14 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*15+10, y=0.41, label="171×15 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*16+10, y=0.41, label="171×16 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*17+10, y=0.41, label="171×17 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*18+10, y=0.41, label="171×18 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*19+10, y=0.41, label="171×19 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*20+10, y=0.41, label="171×20 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*21+10, y=0.41, label="171×21 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*22+10, y=0.41, label="171×22 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*23+10, y=0.41, label="171×23 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*24+10, y=0.41, label="171×24 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*25+10, y=0.41, label="171×25 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*26+10, y=0.41, label="171×26 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*27+10, y=0.41, label="171×27 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*28+10, y=0.41, label="171×28 bp", size=5, color="red", ha="left") +
    annotate("text", x=171*29+10, y=0.41, label="171×29 bp", size=5, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (COLO829BL vs COLO829TB)") +
    theme_minimal() +
    theme(
        figure_size=(13, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_colo829bl_vs_colo829tb_ulont_uid_sv_lag5000")


# %%
# INFO: ST001 Lung ONT
st001_lung_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST001_Lung_ONT/ST001_Lung_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st001_lung_ont_uid_sv, st001_lung_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st001_lung_ont_uid)

st001_lung_ont_uid_sv_pr = pr.PyRanges(
    st001_lung_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st001_lung_ont_uid_sv_pr_st001_flagger_overlap_idx = st001_lung_ont_uid_sv_pr.overlap(st001_flagger)["_idx"].values

st001_lung_ont_uid_sv_filtered = (st001_lung_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st001_lung_ont_uid_sv_pr_st001_flagger_overlap_idx))
    .drop("_idx")) 

st001_lung_ont_uid_sv_filtered_total = st001_lung_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st001_lung_ont_uid_sv_filtered_unit = st001_lung_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st001_lung_ont_uid_sv_filtered_nonunit = st001_lung_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st001_lung_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st001_lung_ont_uid_sv_filtered_unit)
st001_lung_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st001_lung_ont_uid_sv_filtered_nonunit)

st001_lung_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st001_lung_ont_uid_sv_filtered_unit_collapsed, st001_lung_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: ST001 Liver ONT
st001_liver_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST001_Liver_ONT/ST001_Liver_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st001_liver_ont_uid_sv, st001_liver_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st001_liver_ont_uid)

st001_liver_ont_uid_sv_pr = pr.PyRanges(
    st001_liver_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st001_liver_ont_uid_sv_pr_st001_flagger_overlap_idx = st001_liver_ont_uid_sv_pr.overlap(st001_flagger)["_idx"].values

st001_liver_ont_uid_sv_filtered = (st001_liver_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st001_liver_ont_uid_sv_pr_st001_flagger_overlap_idx))
    .drop("_idx")) 

st001_liver_ont_uid_sv_filtered_total = st001_liver_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st001_liver_ont_uid_sv_filtered_unit = st001_liver_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st001_liver_ont_uid_sv_filtered_nonunit = st001_liver_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st001_liver_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st001_liver_ont_uid_sv_filtered_unit)
st001_liver_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st001_liver_ont_uid_sv_filtered_nonunit)

st001_liver_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st001_liver_ont_uid_sv_filtered_unit_collapsed, st001_liver_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: ST002 Lung ONT
st002_lung_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST002_Lung_ONT/ST002_Lung_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st002_lung_ont_uid_sv, st002_lung_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st002_lung_ont_uid)

st002_lung_ont_uid_sv_pr = pr.PyRanges(
    st002_lung_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st002_lung_ont_uid_sv_pr_st002_flagger_overlap_idx = st002_lung_ont_uid_sv_pr.overlap(st002_flagger)["_idx"].values

st002_lung_ont_uid_sv_filtered = (st002_lung_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st002_lung_ont_uid_sv_pr_st002_flagger_overlap_idx))
    .drop("_idx")) 

st002_lung_ont_uid_sv_filtered_total = st002_lung_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st002_lung_ont_uid_sv_filtered_unit = st002_lung_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st002_lung_ont_uid_sv_filtered_nonunit = st002_lung_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st002_lung_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st002_lung_ont_uid_sv_filtered_unit)
st002_lung_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st002_lung_ont_uid_sv_filtered_nonunit)

st002_lung_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st002_lung_ont_uid_sv_filtered_unit_collapsed, st002_lung_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: ST002 Colon ONT
st002_colon_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST002_Colon_ONT/ST002_Colon_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st002_colon_ont_uid_sv, st002_colon_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st002_colon_ont_uid)

st002_colon_ont_uid_sv_pr = pr.PyRanges(
    st002_colon_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st002_colon_ont_uid_sv_pr_st002_flagger_overlap_idx = st002_colon_ont_uid_sv_pr.overlap(st002_flagger)["_idx"].values

st002_colon_ont_uid_sv_filtered = (st002_colon_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st002_colon_ont_uid_sv_pr_st002_flagger_overlap_idx))
    .drop("_idx")) 

st002_colon_ont_uid_sv_filtered_total = st002_colon_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st002_colon_ont_uid_sv_filtered_unit = st002_colon_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st002_colon_ont_uid_sv_filtered_nonunit = st002_colon_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st002_colon_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st002_colon_ont_uid_sv_filtered_unit)
st002_colon_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st002_colon_ont_uid_sv_filtered_nonunit)

st002_colon_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st002_colon_ont_uid_sv_filtered_unit_collapsed, st002_colon_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: ST003 Brain ONT
st003_brain_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST003_Brain_ONT/ST003_Brain_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st003_brain_ont_uid_sv, st003_brain_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st003_brain_ont_uid)

st003_brain_ont_uid_sv_pr = pr.PyRanges(
    st003_brain_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st003_brain_ont_uid_sv_pr_st003_flagger_overlap_idx = st003_brain_ont_uid_sv_pr.overlap(st003_flagger)["_idx"].values

st003_brain_ont_uid_sv_filtered = (st003_brain_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st003_brain_ont_uid_sv_pr_st003_flagger_overlap_idx))
    .drop("_idx")) 

st003_brain_ont_uid_sv_filtered_total = st003_brain_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st003_brain_ont_uid_sv_filtered_unit = st003_brain_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st003_brain_ont_uid_sv_filtered_nonunit = st003_brain_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st003_brain_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st003_brain_ont_uid_sv_filtered_unit)
st003_brain_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st003_brain_ont_uid_sv_filtered_nonunit)

st003_brain_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st003_brain_ont_uid_sv_filtered_unit_collapsed, st003_brain_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

# INFO: ST004 Brain ONT
st004_brain_ont_uid = pl.read_csv(f"{centroindel_benchmark_dir}/ST004_Brain_ONT/ST004_Brain_ONT_CDRfromPacBio_unitsized-indels.bed", separator="\t")
st004_brain_ont_uid_sv, st004_brain_ont_uid_sv_counts_table = get_sv_table_from_scanCSV(st004_brain_ont_uid)

st004_brain_ont_uid_sv_pr = pr.PyRanges(
    st004_brain_ont_uid_sv
    .with_row_index("_idx")
    .with_columns(
        pl.when(pl.col("indel_type") == "INS")
            .then(pl.col("start") -1)
            .otherwise(pl.col("start"))
            .alias("start")
    )
    .to_pandas()
    .rename(columns={"#chrom": "Chromosome", "start": "Start", "end": "End"})
)

st004_brain_ont_uid_sv_pr_st004_flagger_overlap_idx = st004_brain_ont_uid_sv_pr.overlap(st004_flagger)["_idx"].values

st004_brain_ont_uid_sv_filtered = (st004_brain_ont_uid_sv
    .with_row_index("_idx")
    .filter(~pl.col("_idx").is_in(st004_brain_ont_uid_sv_pr_st004_flagger_overlap_idx))
    .drop("_idx")) 

st004_brain_ont_uid_sv_filtered_total = st004_brain_ont_uid_sv_filtered.filter(
    (pl.col("gc_identity") >= 0.995) &
    (pl.col("aligned_fraction") >= 0.995) &
    (pl.col("min_dist_query") / pl.col("read_length") > 0.1)
    )

st004_brain_ont_uid_sv_filtered_unit = st004_brain_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "true")
)

st004_brain_ont_uid_sv_filtered_nonunit = st004_brain_ont_uid_sv_filtered_total.filter(
    (pl.col("is_unit_sized") == "false")
)

st004_brain_ont_uid_sv_filtered_unit_collapsed = collapse_overlapping_sv(st004_brain_ont_uid_sv_filtered_unit)
st004_brain_ont_uid_sv_filtered_nonunit_collapsed = collapse_overlapping_non_unit_sv(st004_brain_ont_uid_sv_filtered_nonunit)

st004_brain_ont_uid_sv_filtered_unit_total_collapsed = pl.concat([st004_brain_ont_uid_sv_filtered_unit_collapsed, st004_brain_ont_uid_sv_filtered_nonunit_collapsed.drop("has_intra_read_conflict")], how="align").sort(["#chrom", "start", "end"])

_, st001_lung_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st001_lung_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st001_liver_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st001_liver_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st002_lung_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st002_lung_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st002_colon_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st002_colon_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st003_brain_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st003_brain_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)
_, st004_brain_ont_uid_sv_filtered_unit_total_collapsed_counts_table = get_sv_table_from_scanCSV(st004_brain_ont_uid_sv_filtered_unit_total_collapsed, min_sv_length=100)

# INFO: Compute Autocorrelation
ac_st001_liver_ont_uid_sv_max_lag2000 = autocorr_range(st001_liver_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st001_lung_ont_uid_sv_max_lag2000 = autocorr_range(st001_lung_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st002_colon_ont_uid_sv_max_lag2000 = autocorr_range(st002_colon_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st002_lung_ont_uid_sv_max_lag2000 = autocorr_range(st002_lung_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st003_brain_ont_uid_sv_max_lag2000 = autocorr_range(st003_brain_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)
ac_st004_brain_ont_uid_sv_max_lag2000 = autocorr_range(st004_brain_ont_uid_sv_filtered_unit_total_collapsed_counts_table["count"], max_lag=2000)

# INFO: Overlay all 6 benchmark tissues autocorrelation (lag up to 2,000)
benchmark_sample_order = [
    "ST001 Liver", "ST001 Lung",
    "ST002 Colon", "ST002 Lung",
    "ST003 Brain", "ST004 Brain",
]

ac_benchmark_ont_combined_lag2000 = pl.concat([
    ac_st001_liver_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST001 Liver").alias("sample")),
    ac_st001_lung_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST001 Lung").alias("sample")),
    ac_st002_colon_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST002 Colon").alias("sample")),
    ac_st002_lung_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST002 Lung").alias("sample")),
    ac_st003_brain_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST003 Brain").alias("sample")),
    ac_st004_brain_ont_uid_sv_max_lag2000.with_columns(pl.lit("ST004 Brain").alias("sample")),
]).to_pandas()

ac_benchmark_ont_combined_lag2000["sample"] = pd.Categorical(
    ac_benchmark_ont_combined_lag2000["sample"],
    categories=benchmark_sample_order, ordered=True,
)

benchmark_color_map = {
    "ST001 Liver": "#01befe",
    "ST001 Lung":  "#ffdd00",
    "ST002 Colon": "#ff7d00",
    "ST002 Lung":  "#ff006d",
    "ST003 Brain": "#adff02",
    "ST004 Brain": "#8f00ff",
}

plot = (
    ggplot(ac_benchmark_ont_combined_lag2000, aes(x="lag", y="autocorr", color="sample")) +
    geom_line() +
    scale_x_continuous(labels=comma_format()) +
    scale_color_manual(values=benchmark_color_map) +
    geom_vline(xintercept=171*1, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*2, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*3, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*4, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*5, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*6, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*7, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*8, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*9, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*10, color="red", linetype="dashed", size=0.3) +
    geom_vline(xintercept=171*11, color="red", linetype="dashed", size=0.3) +
    annotate("text", x=171*1+10,  y=0.3, label="171 bp",    size=8, color="red", ha="left") +
    annotate("text", x=171*2+10,  y=0.3, label="171×2 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*3+10,  y=0.3, label="171×3 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*4+10,  y=0.3, label="171×4 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*5+10,  y=0.3, label="171×5 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*6+10,  y=0.3, label="171×6 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*7+10,  y=0.3, label="171×7 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*8+10,  y=0.3, label="171×8 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*9+10,  y=0.3, label="171×9 bp",  size=8, color="red", ha="left") +
    annotate("text", x=171*10+10, y=0.3, label="171×10 bp", size=8, color="red", ha="left") +
    annotate("text", x=171*11+10, y=0.3, label="171×11 bp", size=8, color="red", ha="left") +
    labs(x="Lag between SV length vectors",
         y="Autocorrelation between SV events",
         title="Autocorrelation of SV lengths within Centromeres (Benchmark tissues)") +
    theme_minimal() +
    theme(
        figure_size=(10, 5),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black'),
        axis_text_y=element_text(color='black'),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.5, color='black'),
        axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.5, color='darkgray'),
        legend_title=element_blank(),
    )
)

ggsavefig_and_show(plot, "autocorrelation_plot_benchmark_tissues_ont_combined_uid_sv_lag2000")

# %%

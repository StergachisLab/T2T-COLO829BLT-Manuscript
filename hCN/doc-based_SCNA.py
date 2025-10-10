# %%
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(font="Arial", font_scale=1.15, style="ticks")
matplotlib.rcParams["figure.dpi"] = 300
plt.rc("axes.spines", top=False, right=False)

# %%
# INFO: 100kb Callable Window
dir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq"

colo829bl = pd.read_table(
    f"{dir}/COLO829BL/Depth/100kb_Callable/COLO829BL_DSA_resetmapq.regions.bed.gz",
    header=None,
    sep="\t",
)
colo829bl.columns = ["contig", "start", "end", "size", "cov_bl"]
colo829bl["id"] = colo829bl.apply(
    lambda x: f"{x['contig']}:{x['start']}-{x['end']}", axis=1
)
colo829bl = colo829bl.set_index("id")
colo829bl_cov_median = colo829bl[colo829bl["cov_bl"] > 0]["cov_bl"].median()

colo829tb = pd.read_table(
    f"{dir}/COLO829T_PassageB/Depth/100kb_Callable/COLO829T_PassageB_DSA_resetmapq.regions.bed.gz",
    header=None,
    sep="\t",
)
colo829tb.columns = ["contig", "start", "end", "size", "cov_tb"]
colo829tb["id"] = colo829tb.apply(
    lambda x: f"{x['contig']}:{x['start']}-{x['end']}", axis=1
)
colo829tb = colo829tb.set_index("id")
colo829tb_cov_median = colo829tb[colo829tb["cov_tb"] > 0]["cov_tb"].median()

colo829ta = pd.read_table(
    f"{dir}/COLO829T_PassageA/Depth/100kb_Callable/COLO829T_PassageA_DSA_resetmapq.regions.bed.gz",
    header=None,
    sep="\t",
)
colo829ta.columns = ["contig", "start", "end", "size", "cov_ta"]
colo829ta["id"] = colo829ta.apply(
    lambda x: f"{x['contig']}:{x['start']}-{x['end']}", axis=1
)
colo829ta = colo829ta.set_index("id")
colo829ta_cov_median = colo829ta[colo829ta["cov_ta"] > 0]["cov_ta"].median()

merged_cov_100kb = pd.concat(
    [colo829bl["cov_bl"], colo829tb["cov_tb"], colo829ta["cov_ta"]], axis=1
)

# del colo829bl, colo829tb, colo829ta
# NOTE: Per Mitchell's Calculation
colo829bl_cov_median = 176
# colo829tb_cov_median = 123
colo829tb_cov_median = 61.5
# colo829ta_cov_median

merged_cov_100kb["tb_bl_ratio"] = (
    merged_cov_100kb["cov_tb"] / colo829tb_cov_median
) / (merged_cov_100kb["cov_bl"] / colo829bl_cov_median)
merged_cov_100kb["ta_bl_ratio"] = (
    merged_cov_100kb["cov_ta"] / colo829ta_cov_median
) / (merged_cov_100kb["cov_bl"] / colo829bl_cov_median)

merged_cov_100kb["log2_tb_bl_ratio"] = np.log2(merged_cov_100kb["tb_bl_ratio"])
merged_cov_100kb["log2_ta_bl_ratio"] = np.log2(merged_cov_100kb["ta_bl_ratio"])

outdir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/Structural_Variations/Simple_Segment-based_CNV"

merged_cov_100kb.to_csv(f"{outdir}/merged_cov_callable_100kb_log2ratio.tsv", sep="\t")


# %%
# INFO: 100kb Window (HP1-HP2 CN)
dir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/Structural_Variations/Simple_Segment-based_CNV/HP1_HP2_SCNA"

colo829bl = pd.read_table(
    f"{dir}/COLO829BL/Depth/100kb_Callable/COLO829BL-InjectedInto-Sohny_COLOBL_noChrSplitting.full-SurjOnto-COLO829BL_hap1_and_COLO829BL_hap2_MERGED_sorted.regions.bed.gz",
    header=None,
    sep="\t",
)
colo829bl.columns = ["contig", "start", "end", "size", "cov_bl"]
colo829bl["id"] = colo829bl.apply(
    lambda x: f"{x['contig']}:{x['start']}-{x['end']}", axis=1
)
colo829bl = colo829bl.set_index("id")
colo829bl_cov_median = colo829bl[colo829bl["cov_bl"] > 0]["cov_bl"].median()

colo829tb = pd.read_table(
    f"{dir}/COLO829TB/Depth/100kb_Callable/COLO829T-InjectedInto-Sohny_COLOBL_noChrSplitting.full-SurjOnto-COLO829BL_hap1_and_COLO829BL_hap2_MERGED_sorted.regions.bed.gz",
    header=None,
    sep="\t",
)
colo829tb.columns = ["contig", "start", "end", "size", "cov_tb"]
colo829tb["id"] = colo829tb.apply(
    lambda x: f"{x['contig']}:{x['start']}-{x['end']}", axis=1
)
colo829tb = colo829tb.set_index("id")
colo829tb_cov_median = colo829tb[colo829tb["cov_tb"] > 0]["cov_tb"].median()

merged_cov_100kb = pd.concat([colo829bl["cov_bl"], colo829tb["cov_tb"]], axis=1)

# del colo829bl, colo829tb, colo829ta
# NOTE: Per Mitchell's Calculation
colo829bl_cov_median = 176
# colo829tb_cov_median = 123
colo829tb_cov_median = 61.5
# colo829ta_cov_median

merged_cov_100kb["tb_bl_ratio"] = (
    merged_cov_100kb["cov_tb"] / colo829tb_cov_median
) / (merged_cov_100kb["cov_bl"] / colo829bl_cov_median)

merged_cov_100kb["log2_tb_bl_ratio"] = np.log2(merged_cov_100kb["tb_bl_ratio"])

outdir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/Structural_Variations/Simple_Segment-based_CNV/HP1_HP2_SCNA"

merged_cov_100kb.to_csv(
    f"{outdir}/merged_cov_callable_100kb_hp1-hp2_log2ratio.tsv", sep="\t"
)

# %%
import gzip as gz

import pandas as pd
from plotnine import *


def ggsavefig_and_show(
    plot,
    filename: str,
    plotdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Plots",
    dpi=300,
):
    """
    plot: plotnine.ggplot.ggplot
    """
    plot.save(f"{plotdir}/{filename}.pdf", dpi=dpi)

    os.system(f"code -r {plotdir}/{filename}.pdf")


def read_fai(fai_file):
    contig_lengths = {}
    with open(fai_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            contig = fields[0]
            length = int(fields[1])
            contig_lengths[contig] = length
    return contig_lengths


def read_cn_data(bed_file):
    if bed_file.endswith(".gz"):
        opener = gz.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    with opener(bed_file, mode) as f:
        header = (
            f.readline().strip().split("\t")
        )  # NOTE: merged_cov_callable_100kb_log2ratio_CBS_wCN.bed.gz
        data = []
        for line in f:
            fields = line.strip().split("\t")
            data.append(fields)

    df = pd.DataFrame(data, columns=header)

    numeric_cols = [
        "start",
        "end",
        "cov_bl",
        "cov_tb",
        "cov_ta",
        "tb_bl_ratio",
        "ta_bl_ratio",
        "log2_tb_bl_ratio",
        "log2_ta_bl_ratio",
        "log2_tb_bl_ratio_capped",
        "position",
        "num.mark",
        "seg.mean",
        "seg.median",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_copy_number(bed_file, fai_file, contig_name, chromosome: str, ylim=None):
    """
    Create a copy number plot for a specific contig

    Parameters:
    -----------
    bed_file : str
        Path to the merged_cov_callable_100kb_log2ratio_CBS_wCN.bed.gz file
    fai_file : str
        Path to the FAI file
    contig_name : str
        Name of the contig to plot (e.g., 'haplotype1-0000012')
    chromosome : str
        Name of the chromosome-of-origin for input contig (e.g., 'chr1')
    ylim : tuple, optional
        Y-axis limits for the log2 ratio plot ((ymin,ymax), default is None)
    """

    print(f"Reading copy number data from {bed_file}...")
    cn_data = read_cn_data(bed_file)

    print(f"Reading contig lengths from {fai_file}...")
    contig_lengths = read_fai(fai_file)

    contig_data = cn_data[cn_data["#chromosome"] == contig_name].copy()

    if len(contig_data) == 0:
        raise ValueError(f"No data found for contig {contig_name}")

    print(f"Found {len(contig_data)} data points for {contig_name}")

    if contig_name not in contig_lengths:
        raise ValueError(f"Contig {contig_name} not found in FAI file")

    contig_length = contig_lengths[contig_name]

    segments = []
    current_median = contig_data.iloc[0]["seg.median"]
    segment_start = contig_data.iloc[0]["start"]

    for i in range(1, len(contig_data)):
        if contig_data.iloc[i]["seg.median"] != current_median:
            segment_end = contig_data.iloc[i - 1]["end"]
            segments.append(
                {
                    "x": segment_start,
                    "xend": segment_end,
                    "y": current_median,
                    "yend": current_median,
                }
            )

            current_median = contig_data.iloc[i]["seg.median"]
            segment_start = contig_data.iloc[i]["start"]

    segment_end = contig_data.iloc[-1]["end"]
    segments.append(
        {
            "x": segment_start,
            "xend": segment_end,
            "y": current_median,
            "yend": current_median,
        }
    )

    segments_df = pd.DataFrame(segments)
    print(f"Found {len(segments_df)} segments")

    plot = (
        ggplot(contig_data, aes(x="position", y="log2_tb_bl_ratio_capped"))
        + geom_point(size=0.05, alpha=0.1, color="#556b2f")
        + geom_segment(
            aes(x="x", xend="xend", y="y", yend="yend"),
            data=segments_df,
            color="red",
            size=0.8,
        )
        + scale_x_continuous(
            limits=[0, contig_length], labels=lambda x: [f"{v / 1e6:.0f}" for v in x]
        )
        + labs(
            title=f"log2hCN Profile - {chromosome} ({contig_name})",
            x="Position (Mb)",
            y="Log2 hCN Ratio",
        )
        + theme_light()
        + theme(
            text=element_text(family="Arial"),
            figure_size=(12, 4),
            plot_title=element_text(size=12, color="black", weight="bold"),
            axis_text=element_text(size=10, color="black"),
        )
    )

    if ylim is not None:
        plot = plot + scale_y_continuous(limits=ylim)

    ggsavefig_and_show(plot, f"log2hcn_profile_{contig_name}")


# %%
colo829bl_dsa_fai = (
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0.fasta.fai"
)
input_hcn = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/Structural_Variations/Simple_Segment-based_CNV/merged_cov_callable_100kb_log2ratio_CBS_wCN.bed.gz"

# %%
# INFO: Contig-Chromosome Assignment (using mapping between DSA and T2T-CHM13v2.0) - Youngjun Kwon
dsa_paf = "/mmfs1/gscratch/stergachislab/mhsohny/Tools/asm-to-reference-alignment/results/T2T_chm13/chain/DSA_COLO829BL_v3.0.0_1_2.cat.paf"
paf_header = [
    "query_name",
    "q_len",
    "q_start",
    "q_end",
    "strand",
    "target_name",
    "t_len",
    "t_start",
    "t_end",
    "n_match",
    "block_len",
    "mapq",
    "id",
    "cigar",
]

df_paf_raw = pd.read_table(dsa_paf, sep="\t", header=None, names=paf_header)

df_paf_raw["aligned_length"] = df_paf_raw["q_end"] - df_paf_raw["q_start"]
agg = (
    df_paf_raw.groupby(["query_name", "target_name"])["aligned_length"]
    .sum()
    .reset_index()
)
max_chr = agg.loc[agg.groupby("query_name")["aligned_length"].idxmax()].copy()
max_chr = max_chr.rename(
    columns={
        "target_name": "primary_chromosome",
        "aligned_length": "primary_aligned_length",
    }
)
total_aligned = agg.groupby("query_name")["aligned_length"].sum().reset_index()
total_aligned = total_aligned.rename(columns={"aligned_length": "total_aligned_length"})
query_lengths = df_paf_raw.drop_duplicates("query_name")[["query_name", "q_len"]]
df_paf = max_chr.merge(total_aligned, on="query_name").merge(
    query_lengths, on="query_name"
)
df_paf["primary_pct"] = df_paf["primary_aligned_length"] / df_paf["q_len"] * 100
df_paf["other_pct"] = (
    (df_paf["total_aligned_length"] - df_paf["primary_aligned_length"])
    / df_paf["q_len"]
    * 100
)

df_paf.to_csv(
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_contig_to_t2t-chm13_primary_chromosome_assignment.tsv",
    sep="\t",
    index=False,
)

# %%
# INFO: chr1-hap1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000012",
    chromosome="chr1",
)

# INFO: chr1-hap2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000059",
    chromosome="chr1",
)

# %%
# INFO: chr3-hap1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000011",
    chromosome="chr3",
)

# INFO: chr3-hap2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000058",
    chromosome="chr3",
)

# %%
# INFO: chr4-hap1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000013",
    chromosome="chr4",
)

# INFO: chr4-hap2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000060",
    chromosome="chr4",
)

# %%
# INFO: chr14-hap1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000018",
    chromosome="chr14",
)

# INFO: chr14-hap2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000070",
    chromosome="chr14",
)

# %%
# INFO: chr16-hap1 segment 1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000008",
    chromosome="chr16",
)

# INFO: chr16-hap1 segment 2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype1-0000016",
    chromosome="chr16",
    ylim=(-1, None),
)

# %%
# INFO: chr16-hap2 segment 1
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000055",
    chromosome="chr16",
)

# INFO: chr16-hap2 segment 2
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000046",
    chromosome="chr16",
    ylim=(0, 1),
)

# INFO: chr16-hap2 segment 3
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000067",
    chromosome="chr16",
)

# INFO: chr16-hap2 segment 4
plot_copy_number(
    bed_file=f"{input_hcn}",
    fai_file=f"{colo829bl_dsa_fai}",
    contig_name="haplotype2-0000066",
    chromosome="chr16",
)
# %%

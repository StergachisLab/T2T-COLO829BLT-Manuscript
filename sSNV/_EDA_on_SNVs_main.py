# %%
%%HTML
<style>
    body {
        --vscode-font-family: "CaskaydiaCove Nerd Font"
    }
</style>
# %% [markdown]

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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2, venn3
from plotnine import *
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

def savefig_and_show(
    filename,
    plotdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Plots",
):
    plt.savefig(f"{plotdir}/{filename}.pdf", bbox_inches="tight", dpi=300)

    os.system(f"code -r {plotdir}/{filename}.pdf")

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


def vcf_format_getter(df, field) -> pd.Series:
    """
    Parameters
    ----------

    df : pandas.core.frame.DataFrame
        vcf read through read_vcf()
    field : str
        GT, VAF, DP, AD

    return pd.Series

    Example vcf structure
    ----------
    FORMAT              COLO829T_PassageB_DSA
    GT:GQ:DP:AD:VAF:PL  1/1:59:133:2,131:0.984962:65,60,0
    GT:GQ:DP:AD:VAF:PL  1/1:65:131:0,131:1:70,66,0
    GT:GQ:DP:AD:VAF:PL  1/1:53:49:0,49:1:58,54,0

    """
    sampleid = df.columns[9]
    format = list(set(df["FORMAT"].values))[0].split(":")

    if field == "GT":
        gtindex = format.index("GT")
        return df[sampleid].str.split(":").apply(lambda x: x[gtindex])

    elif field == "VAF":
        vafindex = format.index("VAF")
        return df[sampleid].str.split(":").apply(lambda x: float(x[vafindex]))

    elif field == "DP":
        dpindex = format.index("DP")
        return df[sampleid].str.split(":").apply(lambda x: int(x[dpindex]))

    elif field == "AD":
        adindex = format.index("AD")
        return (
            df[sampleid].str.split(":").apply(lambda x: int(x[adindex].split(",")[1]))
        )

    elif field == "AD_REF":
        adindex = format.index("AD")
        return (
            df[sampleid].str.split(":").apply(lambda x: int(x[adindex].split(",")[0]))
        )

    elif field == "AD_ALL":
        adindex = format.index("AD")
        return (
            df[sampleid].str.split(":").apply(lambda x: int(x[adindex].split(",")[0]) + int(x[adindex].split(",")[1]))
        )


    else:
        raise ValueError("field should be one of GT, VAF, DP and AD!")


def vcf_info_parser(info_string) -> dict:
    """
    Parameters
    ----------
    info_string : str
    SP=STK,RF,MT2,VN;RGN=Difficult;RGN_T=Tier2;VAF_Ill=0.965;VAF_PB=1

    return dict

    """
    return {i.split("=")[0]: i.split("=")[1] for i in info_string.split(";")}


def vcf_info_getter(df, field):
    """
    Parameters
    ----------

    df : pandas.core.frame.DataFrame
        vcf read through read_vcf()
    field : str
        VAF_Ill, VAF_PB

    return pd.Series

    Example vcf structure
    ----------
    INFO
    SP=STK,RF,MT2,VN;RGN=Difficult;RGN_T=Tier2;VAF_Ill=0.965;VAF_PB=1
    SP=STK,RF,MT2,VN;RGN=Easy;RGN_T=Tier0;VAF_Ill=0.981;VAF_PB=0.984
    """

    if field == "VAF_Ill" or field == "VAF_PB":
        return df["INFO"].apply(
            lambda x: float(vcf_info_parser(x)[field])
            if vcf_info_parser(x)[field] != "NA"
            else np.nan
        )

    else:
        return df["INFO"].apply(lambda x: vcf_info_parser(x).get(field))


def make_site_list(df: pd.DataFrame, path: str, prefix: str) -> None:
    """
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        vcf read through read_vcf()
    path : str

    prefix : str

    return target_list
    """

    df = df[["CHROM", "POS", "POS"]].drop_duplicates()

    df.to_csv(
        f"{os.path.join(path, prefix)}.sitelist", sep="\t", index=False, header=False
    )


"""
def get_pileup_nonref_snvid_from_bamreadcount(string):
	'''
    Parameters
    ----------
    haplotype1-0000001      33514324        C       163     =:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00  A:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00  C:163:60.00:37.20:60.00:77:86:0.45:0.00:85.27:77:0.52:21136.30:0.50     G:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00  T:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00  N:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00  +A:1:60.00:0.00:60.00:0:1:0.38:0.01:268.00:0:0.00:23061.00:0.81
    '''
	threshold = 1
	
	string2list = string.strip().split()
	preid = f"{'_'.join(string2list[:3])}_"
	
	snvid_list = list()
	
	for s in string2list[5:9]:
		s = s.split(':')
		if s[0] != string2list[2] and int(s[1]) >= threshold:
			snvid_list.append(preid + s[0])
	
	return snvid_list
"""


def reverse_complement(string):
    try:
        complement_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
        complement_string = "".join([complement_dict[s] for s in string])
    except KeyError:
        raise ValueError("Invalid character other than A,T,G and C")
    return complement_string[::-1]


def forward_fill_coverage_columns(df):
    """
    Forward fill non-empty values in total coverage columns (2,4,6,8,10,12,14)
    2: BL Fiber-seq
    4: TB Fiber-seq
    6: TA Fiber-seq
    """
    filled_df = df.copy()
    coverage_cols = [2, 4, 6]

    for posid in df["POSid"].unique():
        mask = filled_df["POSid"] == posid
        group_data = filled_df[mask]

        for col_idx in coverage_cols:
            if col_idx < len(df.columns):
                non_empty = group_data.iloc[:, col_idx].replace("", pd.NA).dropna()
                if not non_empty.empty:
                    filled_df.loc[mask, filled_df.columns[col_idx]] = non_empty.iloc[0]

    return filled_df


def calculate_fisher_tests(df, fisher_alternative="greater"):
    """
    1,2: BL Fiber-seq
    3,4: TB Fiber-seq
    5,6: TA Fiber-seq
    """
    results = []

    for _, row in df.iterrows():
        fiberseq_bl = [
            int(float(row.iloc[1])),
            int(float(row.iloc[2])) - int(float(row.iloc[1])),
        ]
        fiberseq_tb = [
            int(float(row.iloc[3])),
            int(float(row.iloc[4])) - int(float(row.iloc[3])),
        ]
        fiberseq_ta = [
            int(float(row.iloc[5])),
            int(float(row.iloc[6])) - int(float(row.iloc[5])),
        ]

        fiberseq_tb_table = [fiberseq_tb, fiberseq_bl]
        fiberseq_ta_table = [fiberseq_ta, fiberseq_bl]

        fiberseq_tb_pvalue = fisher_exact(
            fiberseq_tb_table, alternative=fisher_alternative
        ).pvalue
        fiberseq_ta_pvalue = fisher_exact(
            fiberseq_ta_table, alternative=fisher_alternative
        ).pvalue

        results.append(
            {
                "pvalue_Fiber-seq_TB": fiberseq_tb_pvalue,
                "pvalue_Fiber-seq_TA": fiberseq_ta_pvalue,
            }
        )

    return pd.DataFrame(results)


def apply_bh_correction(df, pvalue_cols: list[str], alpha: float=0.05) -> pd.DataFrame:
    df_corrected = df.copy()
    for col in pvalue_cols:
        if col in df_corrected.columns:
            pvalues = df_corrected[col]
            rejected, pvals_corrected, _, _ = multitest.multipletests(
                pvalues, alpha=alpha, method="fdr_bh"
            )
            df_corrected[f"{col}_BH"] = pvals_corrected
            df_corrected[f"significant_{col.replace('pvalue_', '')}_BH"] = rejected

    return df_corrected

def vcf_in_pyranges_interval(df: pd.DataFrame, intervals:pr.pyranges_main.PyRanges, id:str="SNVid") -> pd.DataFrame:
    """
    df: pd.DataFrame read by read_vcf()
    interval: pyranges interval of interest
    id: defaults to "SNVid" ("INDELid" for indels)
    NOTE: This function assumes that there are 'SNVid' column present in the df
    """

    id_of_interest = list()

    for _, interval in intervals.df.iterrows():
        mask = (
            (df['CHROM'] == interval['Chromosome']) &
            (df['POS'] > interval['Start']) &
            (df['POS'] <= interval['End'])
        )

        id_of_interest.extend(df[mask][id].values)

    return df[df[id].isin(id_of_interest)].copy().reset_index(drop=True)

def make_vcf_from_read_vcf(df: pd.DataFrame, prefix: str, outdir: str) -> None:

    dir = '/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq'

    # INFO: Generate Pseudo-VCF Header
    os.system(
        f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {outdir}/pseudovcf_header"
        )
    

    pre_vcf = df.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
    pre_vcf.to_csv(
        f"{outdir}/{prefix}.pre.vcf",
        sep="\t",
        index=False,
        )
    
    # INFO: Combine Pseudo VCF Header with Pre VCF 
    os.system(
        f"cat {outdir}/pseudovcf_header \
            {outdir}/{prefix}.pre.vcf \
            > {outdir}/{prefix}.vcf"
        )

    # INFO:  Compress and Index VCF
    os.system(
        f"bgzip -f {outdir}/{prefix}.vcf \
          && tabix -p vcf {outdir}/{prefix}.vcf.gz"
        )

    # INFO: Remove Pre VCF
    os.system(
        f"rm {outdir}/{prefix}.pre.vcf"
        )

def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def get_kmer_frequency_from_interval(intervals: pr.pyranges_main.PyRanges, 
                                     reference: str = '/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0.fasta', 
                                     k: int=3) -> pd.DataFrame:
    """
    The function is particularly designed to generate the 3-mer normalization factors
    3-mer is based on SBS96 trinucleotides which middle base is one of two pyrimidines (C or T)
    """

    temp_dir = "/mmfs1/gscratch/stergachislab/mhsohny/temp"
    tempf_prefix = random_string(10)

    # INFO: Make Temp BED file from pyranges object
    intervals.to_bed(f'{temp_dir}/{tempf_prefix}.bed')

    # INFO: rustybam get-fasta
    os.system(
        f"rb get-fasta \
            --fasta {reference} \
            --bed {temp_dir}/{tempf_prefix}.bed \
            > {temp_dir}/{tempf_prefix}.fasta"
        )

    # INFO: Canonical K-mer Frequency using jellyfish
    # NOTE: Count
    os.system(
        f"jellyfish count \
            -m {k} \
            --canonical \
            -s 3G -t 4 \
            {temp_dir}/{tempf_prefix}.fasta \
            --output {temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.jf"
        )

    # NOTE: Dump
    os.system(
        f"jellyfish dump \
            {temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.jf \
            > {temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.txt"
        )


    # INFO: Generate K-mer Frequency table
    kmer_canonical_dict = dict() # NOTE: Represented as Canonical K-mer
    with open(f"{temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.txt", 'r') as dfh:
        c = 1
        for line in dfh:
            if c % 2 == 1:
                kmer_count = int(line.strip().lstrip(">"))
            elif c % 2 == 0:
                kmer_canonical_dict[line.strip()] = kmer_count
            c += 1

    kmer_canonical_pyrimidine_dict = dict() # NOTE: Particularly designed for SBS96 signatures
    for kmer, count in kmer_canonical_dict.items():
        middle_base_index = k // 2
        if kmer[middle_base_index] not in ["C", "T"]:
            kmer_canonical_pyrimidine_dict[reverse_complement(kmer)] = count
        else:
            kmer_canonical_pyrimidine_dict[kmer] = count

    kmer_canonical_pyrimidine_dict = dict(sorted(kmer_canonical_pyrimidine_dict.items()))
    kmer_canonical_pyrimidine_df = pd.DataFrame([kmer_canonical_pyrimidine_dict]).T
    kmer_canonical_pyrimidine_df.index.name = f"{k}mer"
    kmer_canonical_pyrimidine_df.columns = ["count"]

    """
    os.system(f"rm -f \
              {temp_dir}/{tempf_prefix}.bed \
              {temp_dir}/{tempf_prefix}.fasta \
              {temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.jf \
              {temp_dir}/{tempf_prefix}.fasta.{k}mers_canonical.txt")
    """

    return kmer_canonical_pyrimidine_df

def get_kmer_fraction_from_frequency_tab(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """
    The function calculates the fraction of each K-mer in the frequency table from get_kmer_frequency_from_interval() function.
    """
    return kmer_df.div(kmer_df.sum(axis=0), axis=1)

def get_3mer_norm_factor(input_df: pd.DataFrame, bg_df: pd.DataFrame) -> pd.DataFrame:
    """
    The function is particularly designed to generate the 3-mer normalization factors
    3-mer is based on SBS96 trinucleotides which middle base is one of two pyrimidines (C or T)

    input_df: pd.DataFrame from get_kmer_frequency_from_interval()
    bg_df: pd.DataFrame from get_kmer_frequency_from_interval() # NOTE: K-mer Fraction for Whole Genomes Assayed (e.g., df_canonical_sbs96_fraction)

    Both DataFrames should have same 'count' columns which have the fraction for each K-mer

    Basically, these series of functions are only limited to 3-mers for now.
    """

    return bg_df['count'] / input_df['count']

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_tuple[0] * 255),
        int(rgb_tuple[1] * 255),
        int(rgb_tuple[2] * 255)
    )


dir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq"

DSA = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0.fasta"

# INFO:
sbs_color_dict = {'SBS2': "#fcf5b4",
                  'SBS7a': '#f0e68c',
                  'SBS7b': '#ffd700',
                  'SBS7c': '#eeb900',
                  'SBS7d': '#dd9c00',
                  'SBS38': '#cc7f00',
                  'SBS97': '#ba6101',
                  'DBS1': '#800000',
                  'SBS13': '#1f77b4',
                  'SBS17a': '#ff7f0e',
                  'SBS17b': '#2ca02c',
                   'SBS57': '#98df8a',
                   'SBS53': '#d62728',
                   'SBS5': '#ff9896',
                   'SBS40': '#c5b0d5',
                   'SBS31': '#9467bd',
                   'SBS19': '#c49c94',
                   'SBS11': '#e377c2',
                   'SBS10b': '#f7b6d2',
                   'SBS10a': '#7f7f7f',
                   'SBS1': '#c7c7c7'}


# INFO: Callable Regions (None-Flagger-NucFlag) + None-DEL Regions
callable_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_Flagger-NucFlag_100kb-DEL_removed.bed.gz", header=None, sep="\t")
callable_pr = pr.from_dict({
    'Chromosome': callable_bed.iloc[:, 0],
    'Start': callable_bed.iloc[:, 1],
    'End': callable_bed.iloc[:, 2]
})

# INFO: RepeatMasker
repeatmasker_bed = pd.read_table(
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Rhodonite/RepeatMasker/RM_DSA_COLO829BL_v3.0.0_sorted.bed.gz", 
    header=None, sep="\t")

#repeatmasker_pr = pr.from_dict({
#    'Chromosome': repeatmasker_bed.iloc[:, 0],
#    'Start': repeatmasker_bed.iloc[:, 1],
#    'End': repeatmasker_bed.iloc[:, 2],
#    'Name': repeatmasker_bed.iloc[:, 6]
#})

repeatmasker_pr = pr.from_dict({
    'Chromosome': repeatmasker_bed.iloc[:, 0],
    'Start': repeatmasker_bed.iloc[:, 1],
    'End': repeatmasker_bed.iloc[:, 2],
    'Name1': repeatmasker_bed.iloc[:, 6],
    'Name2': repeatmasker_bed.iloc[:, 7],
    'Name3': repeatmasker_bed.iloc[:, 3]
})

del repeatmasker_bed

repeatmasker_pr = repeatmasker_pr.intersect(callable_pr)
repeatmasker_pr_cluster = repeatmasker_pr.cluster(count=True)

# INFO: DupMasker (Segmental Duplications)
dupmasker_bed = pd.read_table(
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Rhodonite/DupMasker/duplicons_DSA_COLO829BL_v3.0.0.bed.gz",
    sep="\t"
)

#haplotype1-0000001      13101   13744   SD9076  0       -       13101   13744   190,190,190

dupmasker_pr = pr.from_dict({
    'Chromosome': dupmasker_bed.iloc[:, 0],
    'Start': dupmasker_bed.iloc[:, 1],
    'End': dupmasker_bed.iloc[:, 2],
    'Name': dupmasker_bed.iloc[:, 3]
})

dupmasker_pr = dupmasker_pr.intersect(callable_pr).merge()

repeatmasker_simple_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Rhodonite/RepeatMasker/RM_simple.bed.gz", header=None, sep="\t")

repeatmasker_simple_pr = pr.from_dict({
    'Chromosome': repeatmasker_simple_bed.iloc[:, 0],
    'Start': repeatmasker_simple_bed.iloc[:, 1],
    'End': repeatmasker_simple_bed.iloc[:, 2],
    'Name': repeatmasker_simple_bed.iloc[:, 3]
})

del repeatmasker_simple_bed

# %%
# INFO: Contig-Chromosome Assignment (using mapping between DSA and T2T-CHM13v2.0) - Youngjun Kwon
dsa_fai = "/mmfs1/gscratch/stergachislab/assemblies/DSA_COLO829BL_v3.0.0.fasta.fai"
dsa_paf = "/mmfs1/gscratch/stergachislab/mhsohny/Tools/asm-to-reference-alignment/results/T2T_chm13/chain/DSA_COLO829BL_v3.0.0_1_2.cat.paf"

fai_header = ["query_name", "seq_length", "offset","linebases", "linewidth"]
paf_header = ["query_name","q_len","q_start","q_end","strand","target_name","t_len","t_start","t_end","n_match","block_len","mapq", "id", "cigar"]

df_contig = pd.read_table(dsa_fai, sep="\t", header=None, names=fai_header)
df_paf_raw = pd.read_table(dsa_paf, sep="\t", header=None, names=paf_header)

df_paf_raw["aligned_length"] = df_paf_raw["q_end"] - df_paf_raw["q_start"]
agg = df_paf_raw.groupby(["query_name", "target_name"])["aligned_length"].sum().reset_index()
max_chr = agg.loc[agg.groupby("query_name")["aligned_length"].idxmax()].copy()
max_chr = max_chr.rename(columns={
    "target_name": "primary_chromosome",
    "aligned_length": "primary_aligned_length"
})
total_aligned = agg.groupby("query_name")["aligned_length"].sum().reset_index()
total_aligned = total_aligned.rename(columns={"aligned_length": "total_aligned_length"})
query_lengths = df_paf_raw.drop_duplicates("query_name")[["query_name", "q_len"]]
df_paf = max_chr.merge(total_aligned, on="query_name").merge(query_lengths, on="query_name")
df_paf["primary_pct"] = df_paf["primary_aligned_length"] / df_paf["q_len"] * 100
df_paf["other_pct"] = (df_paf["total_aligned_length"] - df_paf["primary_aligned_length"]) / df_paf["q_len"] * 100


# %%
# INFO: BUSCO comparison (T2T-CHM13v2.0 vs GRCH38 vs Our DSA hap1 and hap2)
busco_comparison_df = pd.DataFrame({"Category": ["Single copy", "Multi copy", "Fragmented", "Missing"], "GRCh38": [13616, 105, 47, 12], "T2T-CHM13v2.0": [13622, 99, 47, 12], "DSA hap1": [13621, 95, 47, 17], "DSA hap2": [13205, 93, 49, 433]})

df_melted = busco_comparison_df.melt(id_vars='Category', 
                                      var_name='Assembly', 
                                      value_name='Count')

totals = df_melted.groupby('Assembly')['Count'].sum().reset_index()
totals.columns = ['Assembly', 'Total']

df_melted = df_melted.merge(totals, on='Assembly')
df_melted['Fraction'] = df_melted['Count'] / df_melted['Total']

df_melted['Assembly'] = pd.Categorical(df_melted['Assembly'], 
                                        categories=['DSA hap1', 'DSA hap2', 'GRCh38', 'T2T-CHM13v2.0'],
                                        ordered=True)

color_palette = {
    'Missing': '#3c4f54',
    'Single copy': '#b34846',
    'Multi copy': '#00a1d5',
    'Fragmented': '#de8f44'
}

df_melted['Category'] = pd.Categorical(df_melted['Category'], 
                                        categories=['Single copy', 'Multi copy', 'Fragmented', 'Missing'])

plot = (
    ggplot(df_melted, aes(x='Assembly', y='Fraction', fill='Category')) +
    geom_col(position='stack', width=0.7) +
    scale_x_discrete(limits=['T2T-CHM13v2.0', 'GRCh38', 'DSA hap2', 'DSA hap1']) +
    scale_fill_manual(values=color_palette) +
    scale_y_continuous(labels=lambda l: [f'{v:.0%}' for v in l]) +
    labs(x='Assembly', y='Percentage', title='BUSCO Comparison') +
    coord_flip() +
    theme_minimal() +
    theme(
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', ha='center'),
        axis_text_y=element_text(color='black'),
        figure_size=(6, 4),
        legend_position='right'
    )
)

ggsavefig_and_show(plot, "busco_comparison")

# %%
# INFO: Color Palette
uvsig = ['SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS38', 'SBS97', 'DBS1']
uvsig_color_list = ['#f0e68c', '#ffd700', '#eeb900', '#dd9c00', '#cc7f00', '#ba6101', '#800000']
uvsig_color_dict = dict(zip(uvsig, uvsig_color_list))

tims_uvsig = ['SBS1', 'SBS2', 'SBS5', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS13', 'SBS17a', 'SBS17b', 'SBS38', 'SBS40', 'SBS97']


# %%
colobl_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829BL/deepvariant/COLO829BL.deepvariant.PASS.snv.annot.vcf.gz"
)
colotb_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz"
)
colota_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.vcf.gz"
)

colobl_vcf["SNVid"] = (
    colobl_vcf[["CHROM", "POS", "REF", "ALT"]].astype(str).apply("_".join, axis=1)
)
colotb_vcf["SNVid"] = (
    colotb_vcf[["CHROM", "POS", "REF", "ALT"]].astype(str).apply("_".join, axis=1)
)
colota_vcf["SNVid"] = (
    colota_vcf[["CHROM", "POS", "REF", "ALT"]].astype(str).apply("_".join, axis=1)
)

colobl_snvs_set = set(colobl_vcf["SNVid"].values)
colotb_snvs_set = set(colotb_vcf["SNVid"].values)
colota_snvs_set = set(colota_vcf["SNVid"].values)

colobl_snvs_correct = colobl_vcf[
    (vcf_info_getter(colobl_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colobl_vcf, "NucFlag").isna())
]
colotb_snvs_correct = colotb_vcf[
    (vcf_info_getter(colotb_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colotb_vcf, "NucFlag").isna())
]
colota_snvs_correct = colota_vcf[
    (vcf_info_getter(colota_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colota_vcf, "NucFlag").isna())
]

colobl_snvs_correct_set = set(colobl_snvs_correct["SNVid"].values)
colotb_snvs_correct_set = set(colotb_snvs_correct["SNVid"].values)
colota_snvs_correct_set = set(colota_snvs_correct["SNVid"].values)

# %%
colobl_snvs_correct = colobl_vcf[
    (vcf_info_getter(colobl_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colobl_vcf, "NucFlag").isna())
]
colotb_snvs_correct = colotb_vcf[
    (vcf_info_getter(colotb_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colotb_vcf, "NucFlag").isna())
]
colota_snvs_correct = colota_vcf[
    (vcf_info_getter(colota_vcf, "Flagger") == "Hap")
    & (vcf_info_getter(colota_vcf, "NucFlag").isna())
]

colobl_snvs_correct_set = set(colobl_snvs_correct["SNVid"].values)
colotb_snvs_correct_set = set(colotb_snvs_correct["SNVid"].values)
colota_snvs_correct_set = set(colota_snvs_correct["SNVid"].values)

snv_flagset_plus_referenceset = colobl_snvs_correct_set.union(colotb_snvs_correct_set).union(colota_snvs_correct_set) # INFO: Set of SNVs required for Precision and Recall Calculation (Benchmarking Flagship Paper)


# %%
sns.set_theme(font="Arial", font_scale=0.6, style="ticks")
matplotlib.rcParams["figure.dpi"] = 200
plt.rc("axes.spines", top=False, right=False)

total = len(
    (colotb_snvs_correct_set.union(colota_snvs_correct_set)).union(
        colobl_snvs_correct_set
    )
)
venn = venn3(
    [colotb_snvs_correct_set, colota_snvs_correct_set, colobl_snvs_correct_set],
    ("COLO829\n(Passage B)", "COLO829\n(Passage A)", "COLO829BL"),
    subset_label_formatter=lambda x: f"{x:,}\n({(x / total):.2%})",
)

venn.get_patch_by_id("100").set_color("red")
venn.get_patch_by_id("010").set_color("blue")
venn.get_patch_by_id("001").set_color("green")
venn.get_patch_by_id("110").set_color("purple")
venn.get_patch_by_id("101").set_color("yellow")
venn.get_patch_by_id("011").set_color("cyan")
venn.get_patch_by_id("111").set_color("white")


for i in ["100", "010", "001", "110", "101", "011", "111"]:
    venn.get_patch_by_id(i).set_edgecolor("none")
    venn.get_patch_by_id(i).set_alpha(0.4)

sns.set_theme(font="Arial", font_scale=1.15, style="ticks")
matplotlib.rcParams["figure.dpi"] = 300
plt.rc("axes.spines", top=False, right=False)

savefig_and_show("snv_venn_diagram_BLandTBTA")

# %%
sns.set_theme(font="Arial", font_scale=1.0, style="ticks")
matplotlib.rcParams["figure.dpi"] = 200
plt.rc("axes.spines", top=False, right=False)

total = len(
    (
        (colotb_snvs_correct_set.difference(colobl_snvs_correct_set)).union(
            colota_snvs_correct_set.difference(colobl_snvs_correct_set)
        )
    )
)
venn = venn2(
    [
        colotb_snvs_correct_set.difference(colobl_snvs_correct_set),
        colota_snvs_correct_set.difference(colobl_snvs_correct_set),
    ],
    ("Passage B", "Passage A"),
    subset_label_formatter=lambda x: f"{x:,}\n({(x / total):.2%})",
)
venn.get_patch_by_id("10").set_color("red")
venn.get_patch_by_id("01").set_color("blue")
venn.get_patch_by_id("11").set_color("purple")

for i in ["10", "01", "11"]:
    venn.get_patch_by_id(i).set_edgecolor("none")
    venn.get_patch_by_id(i).set_alpha(0.4)

savefig_and_show("snv_venn_diagram_TBTA")

sns.set_theme(font="Arial", font_scale=1.15, style="ticks")
matplotlib.rcParams["figure.dpi"] = 300
plt.rc("axes.spines", top=False, right=False)


# %%
# INFO: Output VCF of 1,492 COLO829BL-only variants
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829BL/deepvariant/COLO829BL.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header"
)

colobl_snvs_correct_blonly = colobl_snvs_correct[
    colobl_snvs_correct["SNVid"].isin(
        colobl_snvs_correct_set.difference(
            colotb_snvs_correct_set.union(colota_snvs_correct_set)
        )
    )
].reset_index(drop=True)

colobl_snvs_correct_blonly_set = set(colobl_snvs_correct_blonly["SNVid"])

pre_vcf = colobl_snvs_correct_blonly.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829BL.deepvariant.PASS.snv.annot.blonly.pre.vcf"
)


# %%
# INFO: BL only Variants Pileup sitelist Creation
make_site_list(
    colobl_snvs_correct_blonly,
    path=f"{dir}/VariantCalls_DeepVariant_1.6.1",
    prefix="COLO829BL_SNVs_Somatic_PileupCheck",
)

# INFO: Use epigerust variant counter to get the allelic count

# %%
colobl_blonly_pileup = pd.read_table(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829BL_DSA_blonly.tsv", sep="\t"
)
colotb_blonly_pileup = pd.read_table(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA_blonly.tsv", sep="\t"
)
colota_blonly_pileup = pd.read_table(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA_blonly.tsv", sep="\t"
)

colobl_blonly_pileup.columns = [
    "SNVid",
    "ALTCOV_COLO829BL_Fiber-seq",
    "TOTCOV_COLO829BL_Fiber-seq",
]
colotb_blonly_pileup.columns = [
    "SNVid",
    "ALTCOV_COLO829TB_Fiber-seq",
    "TOTCOV_COLO829TB_Fiber-seq",
]
colota_blonly_pileup.columns = [
    "SNVid",
    "ALTCOV_COLO829TA_Fiber-seq",
    "TOTCOV_COLO829TA_Fiber-seq",
]

colobl_blonly_pileup = colobl_blonly_pileup.set_index("SNVid")
colotb_blonly_pileup = colotb_blonly_pileup.set_index("SNVid")
colota_blonly_pileup = colota_blonly_pileup.set_index("SNVid")

merged_blonly_pileup = pd.concat(
    [colobl_blonly_pileup, colotb_blonly_pileup, colota_blonly_pileup], axis=1
).sort_index()

merged_blonly_pileup = merged_blonly_pileup.reset_index()
merged_blonly_pileup["POSid"] = merged_blonly_pileup["SNVid"].map(
    lambda x: "_".join(x.split("_")[:2])
)

del colobl_blonly_pileup, colotb_blonly_pileup, colota_blonly_pileup

filled_df = forward_fill_coverage_columns(merged_blonly_pileup)

filled_df = filled_df.fillna(0)
filled_df = filled_df[
    filled_df["SNVid"].isin(colobl_snvs_correct_blonly_set)
].reset_index(drop=True)

fisher_results = calculate_fisher_tests(filled_df, fisher_alternative="less")
fisher_df = pd.concat([filled_df, fisher_results], axis=1)

pvalue_columns = ["pvalue_Fiber-seq_TB", "pvalue_Fiber-seq_TA"]

fisher_blonly_bh = apply_bh_correction(fisher_df, pvalue_columns)

fisher_blonly_bh.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/merged_blonly_pileup_final.tsv",
    sep="\t",
    index=False,
)

del filled_df, fisher_results

# %%
"""
colobl_snvs_correct_blonly_p0_001 = colobl_snvs_correct_blonly[
    colobl_snvs_correct_blonly["SNVid"].isin(
        set(
            fisher_blonly_bh[fisher_blonly_bh["pvalue_Fiber-seq_TB_BH"] < 0.001][
                "SNVid"
            ].values
        )
        & set(
            fisher_blonly_bh[fisher_blonly_bh["pvalue_Fiber-seq_TA_BH"] < 0.001][
                "SNVid"
            ].values
        )
    )
].reset_index(drop=True)

# colotb_fiberseq_indels_correct_fbonly_p0_001.columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "COLO829T", "INDELid"]
"""

# %%
# INFO: Candidate pileup-check sites
colot_snvs_gl_pileupcheck_set = colotb_snvs_set.union(colota_snvs_set).difference(
    colobl_snvs_set
)

colotb_snvs_glfilt = colotb_snvs_correct[
    colotb_snvs_correct["SNVid"].isin(colot_snvs_gl_pileupcheck_set)
]

make_site_list(
    colotb_snvs_glfilt,
    path=f"{dir}/VariantCalls_DeepVariant_1.6.1",
    prefix="COLO829TB_SNVs_Germline_PileupCheck",
)

colota_snvs_glfilt = colota_snvs_correct[
    colota_snvs_correct["SNVid"].isin(colot_snvs_gl_pileupcheck_set)
]

make_site_list(
    colota_snvs_glfilt,
    path=f"{dir}/VariantCalls_DeepVariant_1.6.1",
    prefix="COLO829TA_SNVs_Germline_PileupCheck",
)

os.system(
    f"cat \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829TB_SNVs_Germline_PileupCheck.sitelist \
            {dir}/VariantCalls_DeepVariant_1.6.1/COLO829TA_SNVs_Germline_PileupCheck.sitelist \
                | sort | uniq | bedtools sort -i - > {dir}/VariantCalls_DeepVariant_1.6.1/COLO829TBA_SNVs_Germline_PileupCheck.sitelist"
)

os.system(
    f"rm \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829TB_SNVs_Germline_PileupCheck.sitelist \
            {dir}/VariantCalls_DeepVariant_1.6.1/COLO829TA_SNVs_Germline_PileupCheck.sitelist"
)

# %%
# open up the bcftools mpileup vcf and filter out
# Convert test2.vcf.gz to COLO829BL_Pileup_on_COLO829TBA_SNVs_Germline_PileupCheck.sitelist.norm.vcf.gz
colobl_pileupglfilt_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829BL_Pileup_on_COLO829TBA_SNVs_Germline_PileupCheck.sitelist.norm.reheader.vcf.gz"
)
colobl_pileupglfilt_vcf["SNVid"] = (
    colobl_pileupglfilt_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

colobl_snvs_gl_pileupcheck_set = set(colobl_pileupglfilt_vcf["SNVid"])
colobl_snvs_gl_pileupcheck_lenient_set = set(
    colobl_pileupglfilt_vcf[vcf_format_getter(colobl_pileupglfilt_vcf, "AD") > 1][
        "SNVid"
    ]
)
# %%
bl_pileup_cov1 = set(
    colobl_pileupglfilt_vcf[vcf_format_getter(colobl_pileupglfilt_vcf, "AD") == 1][
        "SNVid"
    ].values
)
bl_pileup_cov2 = set(
    colobl_pileupglfilt_vcf[vcf_format_getter(colobl_pileupglfilt_vcf, "AD") == 2][
        "SNVid"
    ].values
)
bl_pileup_cov3 = set(
    colobl_pileupglfilt_vcf[vcf_format_getter(colobl_pileupglfilt_vcf, "AD") == 3][
        "SNVid"
    ].values
)
bl_pileup_cov4 = set(
    colobl_pileupglfilt_vcf[vcf_format_getter(colobl_pileupglfilt_vcf, "AD") == 4][
        "SNVid"
    ].values
)

tb_bl_pileup_cov1 = set(
    colotb_snvs_correct[colotb_snvs_correct["SNVid"].isin(bl_pileup_cov1)][
        "SNVid"
    ].values
)
ta_bl_pileup_cov1 = set(
    colota_snvs_correct[colota_snvs_correct["SNVid"].isin(bl_pileup_cov1)][
        "SNVid"
    ].values
)
tba_bl_pileup_cov1 = tb_bl_pileup_cov1.intersection(ta_bl_pileup_cov1)
colotb_snvs_bl_pileup_cov1_shared = colotb_snvs_correct[
    colotb_snvs_correct["SNVid"].isin(tba_bl_pileup_cov1)
].reset_index(drop=True)

tb_bl_pileup_cov2 = set(
    colotb_snvs_correct[colotb_snvs_correct["SNVid"].isin(bl_pileup_cov2)][
        "SNVid"
    ].values
)
ta_bl_pileup_cov2 = set(
    colota_snvs_correct[colota_snvs_correct["SNVid"].isin(bl_pileup_cov2)][
        "SNVid"
    ].values
)
tba_bl_pileup_cov2 = tb_bl_pileup_cov2.intersection(ta_bl_pileup_cov2)
colotb_snvs_bl_pileup_cov2_shared = colotb_snvs_correct[
    colotb_snvs_correct["SNVid"].isin(tba_bl_pileup_cov2)
].reset_index(drop=True)

tb_bl_pileup_cov3 = set(
    colotb_snvs_correct[colotb_snvs_correct["SNVid"].isin(bl_pileup_cov3)][
        "SNVid"
    ].values
)
ta_bl_pileup_cov3 = set(
    colota_snvs_correct[colota_snvs_correct["SNVid"].isin(bl_pileup_cov3)][
        "SNVid"
    ].values
)
tba_bl_pileup_cov3 = tb_bl_pileup_cov3.intersection(ta_bl_pileup_cov3)
colotb_snvs_bl_pileup_cov3_shared = colotb_snvs_correct[
    colotb_snvs_correct["SNVid"].isin(tba_bl_pileup_cov3)
].reset_index(drop=True)

tb_bl_pileup_cov4 = set(
    colotb_snvs_correct[colotb_snvs_correct["SNVid"].isin(bl_pileup_cov4)][
        "SNVid"
    ].values
)
ta_bl_pileup_cov4 = set(
    colota_snvs_correct[colota_snvs_correct["SNVid"].isin(bl_pileup_cov4)][
        "SNVid"
    ].values
)
tba_bl_pileup_cov4 = tb_bl_pileup_cov4.intersection(ta_bl_pileup_cov4)
colotb_snvs_bl_pileup_cov4_shared = colotb_snvs_correct[
    colotb_snvs_correct["SNVid"].isin(tba_bl_pileup_cov4)
].reset_index(drop=True)

del bl_pileup_cov1, bl_pileup_cov2, bl_pileup_cov3, bl_pileup_cov4
del tb_bl_pileup_cov1, ta_bl_pileup_cov1, tba_bl_pileup_cov1
del tb_bl_pileup_cov2, ta_bl_pileup_cov2, tba_bl_pileup_cov2
del tb_bl_pileup_cov3, ta_bl_pileup_cov3, tba_bl_pileup_cov3
del tb_bl_pileup_cov4, ta_bl_pileup_cov4, tba_bl_pileup_cov4

# %%
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header"
)

# INFO: COLO829T Passage B
# INFO: Shared Variants
# NOTE: COV1
pre_vcf = colotb_snvs_bl_pileup_cov1_shared.iloc[:, :10].rename(
    columns={"CHROM": "#CHROM"}
)
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov1.pre.vcf"
)

# NOTE: COV2
pre_vcf = colotb_snvs_bl_pileup_cov2_shared.iloc[:, :10].rename(
    columns={"CHROM": "#CHROM"}
)
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov2.pre.vcf"
)

# NOTE: COV3
pre_vcf = colotb_snvs_bl_pileup_cov3_shared.iloc[:, :10].rename(
    columns={"CHROM": "#CHROM"}
)
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov3.pre.vcf"
)

# NOTE: COV4
pre_vcf = colotb_snvs_bl_pileup_cov4_shared.iloc[:, :10].rename(
    columns={"CHROM": "#CHROM"}
)
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.blcov4.pre.vcf"
)


# %%
# NOTE: CHECK the mut spectrum of the above vcfs
raw_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_BLCOV*_Shared_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

raw_files_dataframes = []
for raw_file in raw_files:
   df = pd.read_csv(raw_file, sep='\t', index_col=0).T
   raw_files_dataframes.append(df)

raw_assignment_table = pd.concat(raw_files_dataframes, axis=1)

raw_assignment_table_fraction = raw_assignment_table.div(raw_assignment_table.sum(axis=0), axis=1)
raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction[(raw_assignment_table_fraction != 0).any(axis=1)]

raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction_nonzero.reset_index().rename(columns={'index': 'Samples'})


desired_order = ['SBS1', 'SBS5', 'SBS7a', 'SBS7b', 'SBS30', 'SBS37', 'SBS47', 'SBS54', 'SBS58', 'SBS85'][::-1]
df_long = raw_assignment_table_fraction_nonzero.melt(id_vars='Samples', var_name='Sample_Type', value_name='Value')
df_long['Samples'] = pd.Categorical(df_long['Samples'], categories=desired_order, ordered=True)

plot = (
    ggplot(df_long, aes(x='Sample_Type', y='Samples', fill='Value')) +
    geom_tile(color='white', size=0.5) +
    scale_fill_cmap(cmap_name='inferno', guide=guide_colorbar(nbin=10)) +
    theme_minimal() +
    theme(
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', hjust=0.5),
        axis_text_y=element_text(color='black'),
        figure_size=(10, 6)
    ) +
    labs(
        x='',
        y='SBS Signature',
        fill=''
    )
)

ggsavefig_and_show(plot, "mutational_spectrum_heatmap_snvs_blcov1to4")

# %%
# INFO: Now use lenient pileup set instead of stringent pileup set (June 2025)
colotb_snvs_final = colotb_snvs_correct[
    (~colotb_snvs_correct["SNVid"].isin(colobl_snvs_set))
    & (~colotb_snvs_correct["SNVid"].isin(colobl_snvs_gl_pileupcheck_lenient_set))
].reset_index(drop=True)
colota_snvs_final = colota_snvs_correct[
    (~colota_snvs_correct["SNVid"].isin(colobl_snvs_set))
    & (~colota_snvs_correct["SNVid"].isin(colobl_snvs_gl_pileupcheck_lenient_set))
].reset_index(drop=True)
colotb_snvs_finalset = set(colotb_snvs_final["SNVid"].values)
colota_snvs_finalset = set(colota_snvs_final["SNVid"].values)

colot_shared_finalset = colotb_snvs_finalset.intersection(colota_snvs_finalset)
colotb_specific_finalset = colotb_snvs_finalset.difference(colot_shared_finalset)
colota_specific_finalset = colota_snvs_finalset.difference(colot_shared_finalset)

colotb_snvs_shared_final = colotb_snvs_final[
    colotb_snvs_final["SNVid"].isin(colot_shared_finalset)
].reset_index(drop=True)
colota_snvs_shared_final = colota_snvs_final[
    colota_snvs_final["SNVid"].isin(colot_shared_finalset)
].reset_index(drop=True)

colotb_snvs_bspecific_final = colotb_snvs_final[
    colotb_snvs_final["SNVid"].isin(colotb_specific_finalset)
].reset_index(drop=True)
colota_snvs_aspecific_final = colota_snvs_final[
    colota_snvs_final["SNVid"].isin(colota_specific_finalset)
].reset_index(drop=True)

# %%
total = len((colotb_snvs_finalset.union(colota_snvs_finalset)))
venn = venn2(
    [colotb_snvs_finalset, colota_snvs_finalset],
    ("COLO829\n(Passage B)", "COLO829\n(Passage A)"),
    subset_label_formatter=lambda x: f"{x:,}\n({(x / total):.2%})",
)
venn.get_patch_by_id("10").set_color("red")
venn.get_patch_by_id("01").set_color("blue")
venn.get_patch_by_id("11").set_color("purple")

for i in ["10", "01", "11"]:
    venn.get_patch_by_id(i).set_edgecolor("none")
    venn.get_patch_by_id(i).set_alpha(0.4)

savefig_and_show("snv_venn_diagram")

# %%
"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
# INFO: COLO829T Passage B
## COLO829T Passage B Shared Variants
sns.histplot(
    vcf_format_getter(
        colotb_snvs_final[colotb_snvs_final["SNVid"].isin(colot_shared_finalset)], "VAF"
    ),
    kde=True,
    bins=100,
    color="purple",
    ax=axes[0],
).set_title("Somatic SNVs Shared between Passage B and A", fontsize=14)

axes[0].set_xlabel("Variant Allele Fraction")
axes[0].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

sns.histplot(
    vcf_format_getter(
        colotb_snvs_final[colotb_snvs_final["SNVid"].isin(colotb_specific_finalset)],
        "VAF",
    ),
    kde=True,
    bins=50,
    color="red",
    ax=axes[1],
).set_title("B-Specific Somatic SNVs", fontsize=14)
## COLO829T Passage A A-Specific Variants
axes[1].set_xlabel("Variant Allele Fraction")
axes[1].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

sns.histplot(
    vcf_format_getter(
        colota_snvs_final[colota_snvs_final["SNVid"].isin(colota_specific_finalset)],
        "VAF",
    ),
    kde=True,
    bins=50,
    color="blue",
    ax=axes[2],
).set_title("A-Specific Somatic SNVs", fontsize=14)
## COLO829T Passage A A-Specific Variants
axes[2].set_xlabel("Variant Allele Fraction")
axes[2].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
plt.savefig(
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Plots/vaf_snvs.pdf",
    bbox_inches="tight",
    dpi=300,
)
"""

# %%
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header"
)

# INFO: COLO829T Passage B
## INFO: All Passage B Variants
pre_vcf = colotb_snvs_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.pre.vcf"
)

## INFO: Shared Variants
pre_vcf = colotb_snvs_shared_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf"
)

## INFO: B-Specific variants
pre_vcf = colotb_snvs_bspecific_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.b-specific.pre.vcf"
)

# INFO: COLO829T Passage A
## INFO: All Passage A Variants
pre_vcf = colota_snvs_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.pre.vcf"
)

## INFO: Shared Variants
pre_vcf = colota_snvs_shared_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.shared.pre.vcf"
)

## INFO: A-Specific variants
pre_vcf = colota_snvs_aspecific_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.pre.vcf",
    sep="\t",
    index=False,
)

os.system(
    f"cat {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/pseudovcf_header \
          {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.pre.vcf \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.final.vcf"
)

os.system(
    f"bgzip -f {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.final.vcf \
          && tabix -p vcf {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.final.vcf.gz"
)

os.system(
    f"rm {dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageA_DSA/deepvariant/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.pre.vcf"
)

# %%
# colotb_snvs_shared_final[~vcf_info_getter(colotb_snvs_shared_final, "FIRE").isna()]


# %%
# INFO: Validation by multiple sequencing technologies (e.g., ONT)
os.system(
    f"mkdir -p {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup"
)

make_site_list(
    colotb_snvs_final,
    path=f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup",
    prefix="COLO829TB_SNVs_final_PileupCheck",
)

make_site_list(
    colota_snvs_final,
    path=f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup",
    prefix="COLO829TA_SNVs_final_PileupCheck",
)

os.system(
    f"cat \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TB_SNVs_final_PileupCheck.sitelist \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TA_SNVs_final_PileupCheck.sitelist \
          | sort | uniq | bedtools sort -i - \
          > {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist"
)

os.system(
    f"rm \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TB_SNVs_final_PileupCheck.sitelist \
          {dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TA_SNVs_final_PileupCheck.sitelist"
)

# NOTE: Run bcftools mpileup on this sitelist

# %%
# INFO: Open up pileup VCFs

colotb_snvs_shared_final_set = set(colotb_snvs_shared_final["SNVid"].values)

# INFO: ONT New Motor Protein
colotba_snv_ont_newmotor_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_ONT.norm.reheader.vcf.gz"
)
colotba_snv_ont_newmotor_pileup_vcf["SNVid"] = (
    colotba_snv_ont_newmotor_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

# INFO: Ultralong ONT
colotba_snv_ulont_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_UL-ONT.norm.reheader.vcf.gz"
)
colotba_snv_ulont_pileup_vcf["SNVid"] = (
    colotba_snv_ulont_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

# INFO: ONT New Motor Protein
colotba_snv_ont_newmotor_mapq1_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_ONT_MAPQ1.norm.reheader.vcf.gz"
)
colotba_snv_ont_newmotor_mapq1_pileup_vcf["SNVid"] = (
    colotba_snv_ont_newmotor_mapq1_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

# INFO: Ultralong ONT
colotba_snv_ulont_mapq1_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_UL-ONT_MAPQ1.norm.reheader.vcf.gz"
)
colotba_snv_ulont_mapq1_pileup_vcf["SNVid"] = (
    colotba_snv_ulont_mapq1_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)


# INFO: Illumina Novaseq
colotba_snv_illumina_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_Illumina.norm.reheader.vcf.gz"
)
colotba_snv_illumina_pileup_vcf["SNVid"] = (
    colotba_snv_illumina_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

# INFO: Element AVITI
colotba_snv_element_pileup_vcf = read_vcf(
    f"{dir}/VariantCalls_DeepVariant_1.6.1/Mutational_Spectrum/Validation_byPileup/COLO829TBA_SNVs_final_PileupCheck.sitelist.COLO829T_Element.norm.reheader.vcf.gz"
)
colotba_snv_element_pileup_vcf["SNVid"] = (
    colotba_snv_element_pileup_vcf[["CHROM", "POS", "REF", "ALT"]]
    .astype(str)
    .apply("_".join, axis=1)
)

# %% [markdown]
# Filter Somatic SNVs using Somatic Copy Number Information

# %%
cnadir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV"

# INFO: pgfbSNVid to SNVid Dictionary (COLO829T Passage B (Shared + B-Specific))
snvid_pgfbsnvid_colotb_dict = dict(
    list(
        map(
            lambda x: tuple(x.split("\t")),
            open(
                f"{cnadir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.final.dict.tsv"
            )
            .read()
            .strip()
            .split("\n"),
        )
    )
)

# INFO: COLO829T Passage B SNVs with CNA
colotb_snvs_final_with_cna = pd.read_csv(
    f"{cnadir}/Intersect_wo_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.final_WITH_merged_cov_callable_100kb_log2ratio_CBS_wCN_simple.bed.gz",
    sep="\t",
)
colotb_snvs_final_with_cna.columns = [
    "chrom",
    "start",
    "end",
    "pgfbSNVid",
    "GQ",
    "strand",
    "segmentchrom",
    "segmentStart",
    "segmentEnd",
    "CN",
    "Overlap",
]

colotb_snvs_final_with_cna["SNVid"] = colotb_snvs_final_with_cna["pgfbSNVid"].map(
    lambda x: snvid_pgfbsnvid_colotb_dict[x]
)

colotb_snvs_final_with_cna["id"] = (
    colotb_snvs_final_with_cna[["chrom", "start", "end"]].astype(str).apply("_".join, axis=1)
)

# INFO: COLO829T Passage B SNVs (Surject-Inject Outputs) with CNA
colotb_snvs_final_pgfb_with_cna = pd.read_csv(
    f"{cnadir}/Intersect_wo_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot_peaks__surj_onto_COLO829BL_hap1-2_withTags_WITH_merged_cov_callable_100kb_log2ratio_CBS_wCN_simple.bed.gz",
    sep="\t",
)
colotb_snvs_final_pgfb_with_cna.columns = [
    "chrom",
    "start",
    "end",
    "pgfbSNVid",
    "HapOfOrigin",
    "strand",
    "segmentchrom",
    "segmentStart",
    "segmentEnd",
    "CN",
    "Overlap",
]

colotb_snvs_final_pgfb_with_cna["SNVid"] = colotb_snvs_final_pgfb_with_cna[
    "pgfbSNVid"
].map(lambda x: snvid_pgfbsnvid_colotb_dict[x])

colotb_snvs_final_pgfb_with_cna["id"] = (
    colotb_snvs_final_pgfb_with_cna[["chrom", "start", "end"]].astype(str).apply("_".join, axis=1)
)

# %%
"""
# INFO: COLO829T Passage A SNVs with CNA
colota_snvs_final_with_cna = pd.read_csv(
    f"{cnadir}/Intersect_wo_COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.final_WITH_merged_cov_callable_100kb_log2ratio_CBS_wCN_simple.bed.gz",
    sep="\t",
)
colota_snvs_final_with_cna.columns = [
    "chrom",
    "start",
    "end",
    "pgfbSNVid",
    "GQ",
    "strand",
    "segmentchrom",
    "segmentStart",
    "segmentEnd",
    "CN",
    "Overlap",
]

colota_snvs_final_with_cna["SNVid"] = colota_snvs_final_with_cna["pgfbSNVid"].map(
    lambda x: snvid_pgfbsnvid_colota_dict[x]
)

# INFO: COLO829T Passage A SNVs (Surject-Inject Outputs) with CNA
colota_snvs_final_pgfb_with_cna = pd.read_csv(
    f"{cnadir}/Intersect_wo_COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific_peaks__surj_onto_COLO829BL_hap1-2_withTags_WITH_merged_cov_callable_100kb_log2ratio_CBS_wCN_simple.bed.gz",
    sep="\t",
)
colota_snvs_final_pgfb_with_cna.columns = [
    "chrom",
    "start",
    "end",
    "pgfbSNVid",
    "HapOfOrigin",
    "strand",
    "segmentchrom",
    "segmentStart",
    "segmentEnd",
    "CN",
    "Overlap",
]

colota_snvs_final_pgfb_with_cna["SNVid"] = colota_snvs_final_pgfb_with_cna[
    "pgfbSNVid"
].map(lambda x: snvid_pgfbsnvid_colota_dict[x])

colota_snvs_final_pgfb_with_cna_duplicated_del = colota_snvs_final_pgfb_with_cna[
    (
        colota_snvs_final_pgfb_with_cna.duplicated(
            subset=["chrom", "start", "end"], keep=False
        )
    )
    & (colota_snvs_final_pgfb_with_cna["CN"] == "DEL")
].reset_index(drop=True)

# INFO: pgfbSNVid to SNVid Dictionary (COLO829T Passage A (Only A-Specific))
snvid_pgfbsnvid_colota_dict = dict(
    list(
        map(
            lambda x: tuple(x.split("\t")),
            open(
                f"{cnadir}/COLO829T_PassageA_DSA.deepvariant.PASS.snv.annot.a-specific.final.dict.tsv"
            )
            .read()
            .strip()
            .split("\n"),
        )
    )
)
"""

# %%
del_filtered_colotb_snvid = list() # NOTE: SNV records in DEL regions that are either overlapped or not overlapped

del_filtered_colotb_snvid.extend(
    colotb_snvs_final_with_cna[colotb_snvs_final_with_cna["CN"] == "DEL"]["SNVid"].values.tolist()
)

# INFO: First remove "DEL" associated SNVs (in pgfbSNVid) from DSG-surjected SNV tables
colotb_snvs_final_pgfb_with_cna = colotb_snvs_final_pgfb_with_cna[
    ~(
        colotb_snvs_final_pgfb_with_cna["pgfbSNVid"].isin(
        colotb_snvs_final_with_cna[colotb_snvs_final_with_cna["CN"] == "DEL"]["pgfbSNVid"].values
        )
    )
    ]

# INFO: COLO829T Passage B SNVs in DEL regions that are not overlapped using DSG approach
# NOTE: Could be real but will not be considered in the analysis

del_filtered_colotb_snvid_nonoverlapped = list() # NOTE: To remove

# NOTE: SNVs not overlapped after surjection and somehow landed on DEL regions (should not be removed)
# NOTE: SNVs that are in DEL regions is already removed prior to this
colotb_snvs_final_pgfb_with_cna_del = colotb_snvs_final_pgfb_with_cna[
    ~(
        colotb_snvs_final_pgfb_with_cna.duplicated(
            subset=["chrom", "start", "end"], keep=False
        )
    )
    & (
        colotb_snvs_final_pgfb_with_cna["CN"] == "DEL"
    )
    & (
        (
            (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype1-")) & 
            (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 1)
            )
        | (
            (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype2-")) & 
            (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 2)
            )
    )
    ]


colotb_snvs_final_pgfb_with_cna = colotb_snvs_final_pgfb_with_cna[
    ~(
        colotb_snvs_final_pgfb_with_cna["pgfbSNVid"].isin(
        colotb_snvs_final_pgfb_with_cna_del["pgfbSNVid"]
        )
    )
]

# %%
# INFO: COLO829T Passage B SNVs in DEL regions that are "overlapped (between haps or in the same haps)" using DSG approach
# NOTE: SNV records in DEL region are highly likely to be false positives

del_filtered_colotb_snvid_overlapped = list()

# NOTE: Might be unnecessary but important to check
colotb_snvs_final_pgfb_with_cna_duplicated_del = colotb_snvs_final_pgfb_with_cna[
    (
        colotb_snvs_final_pgfb_with_cna.duplicated(
            subset=["chrom", "start", "end"], keep=False
        )
    )
    & (colotb_snvs_final_pgfb_with_cna["CN"] == "DEL")
].reset_index(drop=True)

# INFO: Identifying Overlapped SNV records inside the DEL segmenets
colotb_dup_and_del_in_hap1_hap1origin = colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype1-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 1)
    ].reset_index(drop=True)

colotb_dup_and_del_in_hap1_hap2origin = colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype1-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 2)
    ].reset_index(drop=True)


del_filtered_colotb_snvid_overlapped.extend(
    colotb_dup_and_del_in_hap1_hap1origin["SNVid"].values.tolist()
) # NOTE: should be doing nothing in this dataset

del_filtered_colotb_snvid.extend(
    colotb_dup_and_del_in_hap1_hap1origin["SNVid"].values.tolist()
) # NOTE: should be doing nothing in this dataset

colotb_dup_and_del_in_hap2_hap2origin = colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype2-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 2)
    ].reset_index(drop=True)

colotb_dup_and_del_in_hap2_hap1origin = colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype2-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 1)
    ].reset_index(drop=True)

del_filtered_colotb_snvid_overlapped.extend(
    colotb_dup_and_del_in_hap2_hap2origin["SNVid"].values.tolist()
)

del_filtered_colotb_snvid.extend(
    colotb_dup_and_del_in_hap2_hap2origin["SNVid"].values.tolist()
)

del_filtered_colotb_snvid_overlapped = set(del_filtered_colotb_snvid_overlapped) # INFO: For merely checking stats (should be empty in this dataset)


# INFO: Tricky Non-Del Overlapped SNVs
colotb_snvs_final_pgfb_with_cna[(~colotb_snvs_final_pgfb_with_cna["SNVid"].isin(del_filtered_colotb_snvid)) & colotb_snvs_final_pgfb_with_cna.duplicated(subset="id", keep=False)].reset_index(drop=True).to_csv(f"{cnadir}/tricky_non_del_overlapped_snvs.tsv", index=False, sep="\t") # 9,320



# %%
from collections import defaultdict

def group_by_coordinates(filename):

    coordinate_groups = defaultdict(set)
    
    with open(filename, 'r') as file:
        file.readline()
        for line in file:
            line = line.strip('\n').split('\t')
            key = (line[0], line[1], line[2])
            coordinate_groups[key].add(line[3])
    
    return dict(coordinate_groups)

def get_all_snv_sets(filename):
    groups = group_by_coordinates(filename)
    return [snv_set for snv_set in groups.values() if len(snv_set) >= 2]

snv_test_needed_set = get_all_snv_sets(f"{cnadir}/tricky_non_del_overlapped_snvs.tsv") # 4,660 pairs 
snv_test_needed_set = [set(s) for s in set(frozenset(s) for s in snv_test_needed_set)] # NOTE: Unique sets only # 2,327 pairs

# %%
snv_test_needed_set_leftover1 = list() # NOTE: Left for validation using ONT New motor
nondel_filtered_colotb_snvid = list()

# DEBUG:
trigger = 0

# INFO: First start with UL-ONT (Gold standard for handling overlapped SNVs)
for snvs in snv_test_needed_set:

    # DEBUG:
    if snvs == {'SNV_24790', 'SNV_56544'}:
        print("Debugging for SNV_24790 and SNV_56544")
        trigger = 1

    snvs_list = list(snvs)
    
    first_snv = snvs_list[0]
    second_snv = snvs_list[1] 

    # INFO: UL-ONT
    ulont_first_snv = colotba_snv_ulont_pileup_vcf[colotba_snv_ulont_pileup_vcf['SNVid'] == snvid_pgfbsnvid_colotb_dict[first_snv]]
    ulont_second_snv = colotba_snv_ulont_pileup_vcf[colotba_snv_ulont_pileup_vcf['SNVid'] == snvid_pgfbsnvid_colotb_dict[second_snv]]

    # INFO: Case where both SNVs are present
    if not ulont_first_snv.empty and not ulont_second_snv.empty:
        contingency_ulont = [
            [vcf_format_getter(ulont_first_snv, "AD").values[0], vcf_format_getter(ulont_first_snv, "AD_REF").values[0]], 
             [vcf_format_getter(ulont_second_snv, "AD").values[0], vcf_format_getter(ulont_second_snv, "AD_REF").values[0]]
            ]
        
        fisher_result_ulont = fisher_exact(
            contingency_ulont
            )

        #print(contingency_ulont, fisher_result_ulont[0], fisher_result_ulont[1])
        if trigger == 1:
            print(first_snv, second_snv)
            print(contingency_ulont, fisher_result_ulont[0], fisher_result_ulont[1])
            trigger = 0

        if fisher_result_ulont[1] < 0.01:
            if fisher_result_ulont[0] > 1:
                nondel_filtered_colotb_snvid.append(second_snv)

            elif fisher_result_ulont[0] < 1:
                nondel_filtered_colotb_snvid.append(first_snv)

        # DEBUG:
        else:
            snv_test_needed_set_leftover1.append(snvs) # NOTE: Left for validation using ONT New motor

    elif ulont_first_snv.empty and not ulont_second_snv.empty:
        if vcf_format_getter(ulont_second_snv, "AD").values[0] >= 3:
            nondel_filtered_colotb_snvid.append(first_snv)
        else:
            snv_test_needed_set_leftover1.append(snvs) # NOTE: Left for validation using ONT New motor

    elif not ulont_first_snv.empty and ulont_second_snv.empty:
        if vcf_format_getter(ulont_first_snv, "AD").values[0] >= 3:
            nondel_filtered_colotb_snvid.append(second_snv)
        else:
            snv_test_needed_set_leftover1.append(snvs) # NOTE: Left for validation using ONT New motor
    
    elif ulont_first_snv.empty and ulont_second_snv.empty:
        snv_test_needed_set_leftover1.append(snvs) # NOTE: Left for validation using ONT New motor


snv_test_needed_set_leftover2 = list() # NOTE: Not validated also using ONT New motor

# DEBUG:
trigger = 0

# INFO: Second, check New motor protein ONT
for snvs in snv_test_needed_set_leftover1:

    # DEBUG:
    if snvs == {'SNV_24790', 'SNV_56544'}:
        print("Debugging for SNV_24790 and SNV_56544")
        trigger = 1

    snvs_list = list(snvs)
    
    first_snv = snvs_list[0]
    second_snv = snvs_list[1] 

    # INFO: ONT New-motor Protein
    newmotor_first_snv = colotba_snv_ont_newmotor_pileup_vcf[colotba_snv_ont_newmotor_pileup_vcf['SNVid'] == snvid_pgfbsnvid_colotb_dict[first_snv]]
    newmotor_second_snv = colotba_snv_ont_newmotor_pileup_vcf[colotba_snv_ont_newmotor_pileup_vcf['SNVid'] == snvid_pgfbsnvid_colotb_dict[second_snv]]

    # INFO: Case where both SNVs are present
    if not newmotor_first_snv.empty and not newmotor_second_snv.empty:
        contingency_newmotor = [
            [vcf_format_getter(newmotor_first_snv, "AD").values[0], vcf_format_getter(newmotor_first_snv, "AD_REF").values[0]], 
             [vcf_format_getter(newmotor_second_snv, "AD").values[0], vcf_format_getter(newmotor_second_snv, "AD_REF").values[0]]
            ]

        fisher_result_newmotor = fisher_exact(
            contingency_newmotor
        )

        print(first_snv, second_snv)
        print(contingency_newmotor, fisher_result_newmotor[0], fisher_result_newmotor[1])

        if trigger == 1:
            print(first_snv, second_snv)
            print(contingency_newmotor, fisher_result_newmotor[0], fisher_result_newmotor[1])
            trigger = 0

        if fisher_result_newmotor[1] < 0.01:
            if fisher_result_newmotor[0] > 1:
                nondel_filtered_colotb_snvid.append(second_snv)

            elif fisher_result_newmotor[0] < 1:
                nondel_filtered_colotb_snvid.append(first_snv)

        # DEBUG:
        else:
            snv_test_needed_set_leftover2.append(snvs) # NOTE: not validated using both UL-ONT and ONT New motor protein

    elif newmotor_first_snv.empty and not newmotor_second_snv.empty:
        print(first_snv, second_snv)
        print(vcf_format_getter(newmotor_second_snv, "AD").values[0])
        if vcf_format_getter(newmotor_second_snv, "AD").values[0] >= 3:
            nondel_filtered_colotb_snvid.append(first_snv)
        else:
            snv_test_needed_set_leftover2.append(snvs) # NOTE: Not validated also using ONT New motor

    elif not newmotor_first_snv.empty and newmotor_second_snv.empty:
        print(first_snv, second_snv)
        print(vcf_format_getter(newmotor_first_snv, "AD").values[0])
        if vcf_format_getter(newmotor_first_snv, "AD").values[0] >= 3:
            nondel_filtered_colotb_snvid.append(second_snv)
        else:
            snv_test_needed_set_leftover2.append(snvs) # NOTE: Not validated also using ONT New motor

    elif newmotor_first_snv.empty and newmotor_second_snv.empty:
        snv_test_needed_set_leftover2.append(snvs) # NOTE: Not validated also using ONT New motor

# NOTE: snv_test_needed_set_leftover2 contains SNV pairs that are not validated using both UL-ONT and ONT New motor protein (231 pairs)
# NOTE: These SNV pairs will be kept in the final filtered set (not removed) - Could be removed during the Density-based filtering step
overlapped_leftover_snv = pd.DataFrame([list(map(lambda x: snvid_pgfbsnvid_colotb_dict[x], sorted(list(s)))) for s in snv_test_needed_set_leftover2], columns=['SNV1', 'SNV2'])

overlapped_leftover_snv.to_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/Overlapped_leftover_SNV.tsv", sep="\t", index=False)
overlapped_leftover_snv_list = list(chain(*list(overlapped_leftover_snv.values)))

overlapped_leftover_snv_list_df = pd.DataFrame([x.split('\t') for x in list(map(lambda x: f"{x.split('_')[0]}\t{int(x.split('_')[1])-1}\t{x.split('_')[1]}", overlapped_leftover_snv_list))], columns=['Chromosome', 'Start', 'End'])

overlapped_leftover_snv_list_df['SNVid'] = 'SNV_' + (overlapped_leftover_snv_list_df.index + 1).astype(str)
overlapped_leftover_snv_list_df["Length"] = 1 
overlapped_leftover_snv_list_df["Strand"] = "+"

overlapped_leftover_snv_list_df.to_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/Overlapped_leftover_SNV.bed", sep="\t", header=None, index=False)

nondel_filtered_colotb_snvid = set(list(map(lambda x: snvid_pgfbsnvid_colotb_dict[x], nondel_filtered_colotb_snvid)))

# DEBUG: Check for haplotype2-0000079_72168510_G_A

# %% 
# INFO: SCNA-DSG-ONT-based Filtering!!
scna_dsg_filtered_snvid = nondel_filtered_colotb_snvid.union(del_filtered_colotb_snvid)

colotb_snvs_shared_final_filtered = colotb_snvs_shared_final[
    ~(colotb_snvs_shared_final["SNVid"].isin(scna_dsg_filtered_snvid))
    ].reset_index(drop=True)

colotb_snvs_shared_final_filtered_set = set(colotb_snvs_shared_final_filtered["SNVid"].values)

# %%
# INFO: Make VCFs for Total, each RE + None_RE
# INFO: Total SNVs
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted"

pre_vcf = colotb_snvs_shared_final_filtered.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.pre.vcf",
    sep="\t",
    index=False,
)

# NOTE: Generate Pseudo VCF Header
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {outdir}/pseudovcf_header"
)

# NOTE: Combine Pseudo VCF Header with Pre VCF
os.system(
    f"cat {outdir}/pseudovcf_header \
          {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.pre.vcf \
          > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf"
)

# NOTE:  Compress and Index VCF
os.system(
    f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf \
          && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf.gz"
)

# NOTE: Remove Pre VCF
os.system(
    f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.pre.vcf"
)

# INFO: For None_RE
retype = "None_RE"
pre_vcf = colotb_snvs_shared_final_filtered[
    (vcf_info_getter(colotb_snvs_shared_final_filtered, "RM").isna())
    ].iloc[:, :10].rename(columns={"CHROM": "#CHROM"})

pre_vcf.to_csv(f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf", sep="\t", index=False)

os.system(f"cat {outdir}/pseudovcf_header \
          {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf \
            > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf")

# NOTE: Compress and Index VCF
os.system(
    f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf \
          && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf.gz"
)

# NOTE: Remove Pre VCF
os.system(
    f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf"
)

# NOTE: For REs

colotb_snvs_shared_final_filtered_re = colotb_snvs_shared_final_filtered[
    ~(vcf_info_getter(colotb_snvs_shared_final_filtered, "RM").isna())
    ].reset_index(drop=True)

for retype in ['DNA', 'LINE', 'LTR', 'Low_complexity', 'Retroposon', 'SINE', 'Satellite', 'Simple_repeat']:
      pre_vcf = colotb_snvs_shared_final_filtered_re[vcf_info_getter(colotb_snvs_shared_final_filtered_re, "RM").str.contains(retype)].iloc[:, :10]
      pre_vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'COLO829T_PassageB_DSA']
      pre_vcf.to_csv(f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf", sep="\t", index=False)
      
      os.system(f"cat {outdir}/pseudovcf_header \
                {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf \
                  > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf")

      os.system(
          f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf \
                && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.vcf.gz")

      os.system(f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.{retype}.pre.vcf")

# NOTE: Remove Pseudo VCF Header
os.system(
    f"rm {outdir}/pseudovcf_header"
)


# %%
"""
colotb_dup_and_nondel_in_hap1_hap1origin = colotb_snvs_final_pgfb_with_cna[
    (colotb_snvs_final_pgfb_with_cna.duplicated(subset=["chrom", "start", "end"], keep=False)) & 
    (colotb_snvs_final_pgfb_with_cna["CN"] != "DEL") & 
    (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype1-")) &
    (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 1)
    ]

colotb_dup_and_nondel_in_hap1_hap2origin = colotb_snvs_final_pgfb_with_cna[
    (colotb_snvs_final_pgfb_with_cna.duplicated(subset=["chrom", "start", "end"], keep=False)) & 
    (colotb_snvs_final_pgfb_with_cna["CN"] != "DEL") & 
    (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype1-")) &
    (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 2)
    ]

# INFO: Two SNVs in the same haplotype coalesced to one coordinates
colotb_dup_and_nondel_in_hap1_hap1origin[(colotb_dup_and_nondel_in_hap1_hap1origin.duplicated(subset=["chrom", "start", "end"], keep=False))]
colotb_dup_and_nondel_in_hap1_hap2origin[(colotb_dup_and_nondel_in_hap1_hap2origin.duplicated(subset=["chrom", "start", "end"], keep=False))]


colotb_dup_and_nondel_in_hap1_hap1origin[~(colotb_dup_and_nondel_in_hap1_hap1origin.duplicated(subset=["chrom", "start", "end"], keep=False))]
colotb_dup_and_nondel_in_hap1_hap2origin[~(colotb_dup_and_nondel_in_hap1_hap2origin.duplicated(subset=["chrom", "start", "end"], keep=False))]




colotb_dup_and_nondel_in_hap2_hap1origin = colotb_snvs_final_pgfb_with_cna[
    (colotb_snvs_final_pgfb_with_cna.duplicated(subset=["chrom", "start", "end"], keep=False)) & 
    (colotb_snvs_final_pgfb_with_cna["CN"] != "DEL") & 
    (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype2-")) &
    (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 1)
    ]

colotb_dup_and_nondel_in_hap2_hap2origin = colotb_snvs_final_pgfb_with_cna[
    (colotb_snvs_final_pgfb_with_cna.duplicated(subset=["chrom", "start", "end"], keep=False)) & 
    (colotb_snvs_final_pgfb_with_cna["CN"] != "DEL") & 
    (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype2-")) &
    (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 2)
    ]

#colotb_snvs_final_pgfb_with_cna_duplicated_del[(colotb_snvs_final_pgfb_with_cna_duplicated_del["start"] > 127_495_649) & (colotb_snvs_final_pgfb_with_cna_duplicated_del["end"] < 127_666_319)]

colotb_dup_and_cn_in_hap1_hap2_origin = colotb_snvs_final_pgfb_with_cna[
    (colotb_snvs_final_pgfb_with_cna["CN"] != "DEL") & 
    (colotb_snvs_final_pgfb_with_cna.duplicated(subset=["chrom", "start", "end"], keep=False)) & 
    (colotb_snvs_final_pgfb_with_cna["chrom"].str.startswith("haplotype1-")) & 
    (colotb_snvs_final_pgfb_with_cna["HapOfOrigin"] == 2)]
"""



# %%
"""
sns.histplot(vcf_format_getter(colotba_snv_ont_newmotor_pileup_vcf[colotba_snv_ont_newmotor_pileup_vcf["SNVid"].isin(colotb_dup_and_del_in_hap1["SNVid"].values)], "AD"))
savefig_and_show("hist_ontnewmotor_ad")


sns.histplot(vcf_format_getter(colotba_snv_ulont_pileup_vcf[colotba_snv_ulont_pileup_vcf["SNVid"].isin(colotb_dup_and_del_in_hap1["SNVid"].values)], "AD"))
savefig_and_show("hist_ulont_ad")

sns.histplot(vcf_format_getter(colotba_snv_ont_newmotor_pileup_vcf[colotba_snv_ont_newmotor_pileup_vcf["SNVid"].isin(colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype1-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 2)
    ].reset_index(drop=True)
["SNVid"].values)], "AD"))
savefig_and_show("hist_ontnewmotor_assigned_ad")


sns.histplot(vcf_format_getter(colotba_snv_ulont_pileup_vcf[colotba_snv_ulont_pileup_vcf["SNVid"].isin(colotb_snvs_final_pgfb_with_cna_duplicated_del[
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["chrom"].str.startswith("haplotype1-")) & 
    (colotb_snvs_final_pgfb_with_cna_duplicated_del["HapOfOrigin"] == 2)
    ].reset_index(drop=True)
["SNVid"].values)], "AD"))
savefig_and_show("hist_ulont_assigned_ad")
"""

# %%
# INFO: 1kb Sliding windows (with 500bp overlap)
ref_fai = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0.fasta.fai"

sliding_bed = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/DSA_COLO829BL_v3.0.0_1kb_500bp-sliding.bed"
sliding_bed_gz = f"{sliding_bed}.gz"
vcf_file = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf.gz"
intersect_bed = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/Intersect_DSA_COLO829BL_v3.0.0_1kb_500bp-sliding_WITH_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf.bed"
intersect_bed_gz = f"{intersect_bed}.gz"
intersect_above1 = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/Intersect_DSA_COLO829BL_v3.0.0_1kb_500bp-sliding_WITH_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf_above1.bed"
intersect_above0 = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/Intersect_DSA_COLO829BL_v3.0.0_1kb_500bp-sliding_WITH_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf_above0.bed"

#os.system(f'bedtools makewindows -g "{ref_fai}" -w 1000 -s 500 > {sliding_bed}')
#os.system(f'bgzip -f -@ 4 {sliding_bed}')
#os.system(f'tabix -p bed {sliding_bed_gz}')


os.system(f'bedtools intersect -c -a {sliding_bed_gz} -b {vcf_file} > {intersect_bed}')
os.system(f'bgzip -@ 4 -f {intersect_bed}')
os.system(f'tabix -p bed {intersect_bed_gz}')

os.system(f"zcat {intersect_bed_gz} | awk '$4 > 1' > {intersect_above1}")
os.system(f"zcat {intersect_bed_gz} | awk '$4 > 0' > {intersect_above0}")


# %%
snvdensity = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/Intersect_DSA_COLO829BL_v3.0.0_1kb_500bp-sliding_WITH_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.filtered.vcf_above0.bed", header=None, sep="\t")
snvdensity.columns = ['contig', 'start', 'end', 'snv_density']

# %%
plot = (ggplot(snvdensity, aes(x='snv_density')) +
        geom_histogram(bins=50, fill='steelblue', alpha=0.7, color='white') +
        scale_y_log10(labels=lambda x: [f'{int(v)}' for v in x]) +
        labs(title='Distribution of SNV Density',
             x='SNV Density',
             y='Count (log scale)') +
        theme_minimal() +
        theme(figure_size=(10, 6),
              plot_title=element_text(size=14),
              axis_title=element_text(size=12))
)
ggsavefig_and_show(plot, "snvdensity_1kb_sliding_window")

# %%
snvdensity_totest = snvdensity[snvdensity["snv_density"] >= 3]

snvdensity_totest_pr = pr.from_dict({
    'Chromosome': snvdensity_totest['contig'],
    'Start': snvdensity_totest['start'],
    'End': snvdensity_totest['end'],
    'snv_density': snvdensity_totest['snv_density']
})

# NOTE: Using PyRanges merge() to merge overlapping intervals and revert back to pd.DataFrame
snvdensity_totest_pr_merge = snvdensity_totest_pr.merge().df.rename(
    columns = {
        'Chromosome': 'contig',
        'Start': 'start',
        'End': 'end'
    })

vaf_data = list()
density_filter_snvid = list()
density_filter_interval_id = list()

for i, interval in snvdensity_totest_pr_merge.iterrows():
    mask = (
        (colotb_snvs_shared_final_filtered['CHROM'] == interval['contig']) &
        (colotb_snvs_shared_final_filtered['POS'] > interval['start']) &
        (colotb_snvs_shared_final_filtered['POS'] <= interval['end'])
    )
    
    snvs_in_interval = colotb_snvs_shared_final_filtered[mask]

    if len(snvs_in_interval) > 0:  # NOTE: There will always be SNVs in the specified interval
        vaf_snvs_in_interval = vcf_format_getter(snvs_in_interval, "VAF").values
        interval_id = f"{interval['contig']}:{interval['start']}-{interval['end']}"

        for vaf in vaf_snvs_in_interval:
            vaf_data.append({
                'interval_id': interval_id,
                'VAF': vaf
            })

        # DEBUG: 
#        if interval_id == "haplotype1-0000012:129691000-129692500":
#            print(vaf_snvs_in_interval)
#            print(np.sum(vaf_snvs_in_interval < 0.75)) 
#            print(vcf_format_getter(snvs_in_interval, "DP").values)
#            print(np.median(vcf_format_getter(snvs_in_interval, "DP").values))

        dp_threshold = 61.5 - 3*poisson(mu=61.5).std() # NOTE: based on 1n coverage of COLO829TB (3 standard deviation away from mean)

        if np.sum(vaf_snvs_in_interval < 0.75) >= 3:
            density_filter_snvid.extend(snvs_in_interval["SNVid"].values)
            
        elif np.median(vcf_format_getter(snvs_in_interval, "DP").values) < dp_threshold:
            density_filter_snvid.extend(snvs_in_interval["SNVid"].values)
            print(f'{interval_id} {snvs_in_interval.shape[0]} {np.median(vcf_format_getter(snvs_in_interval, "DP").values)}')


density_filter_snvid = set(density_filter_snvid)

# %%

vaf_long_df = pd.DataFrame(vaf_data)

count_summary = (
    vaf_long_df.groupby('interval_id')
    .agg({
        'VAF': ['count', 'max']
    })
    .round(3)
)

count_summary.columns = ['count', 'max_vaf']
count_summary = count_summary.reset_index()

count_summary['text_y'] = count_summary['max_vaf'] + 0.01

plot = (
    ggplot(vaf_long_df, aes(x='interval_id', y='VAF')) +
    geom_point(position=position_jitter(width=0.25, height=0), 
               alpha=0.6, size=2, color='steelblue') +
    geom_text(data=count_summary, 
              mapping=aes(x='interval_id', y='text_y', label='count'),
              size=8, ha='center', va='bottom', color='black') +
    theme_minimal() +
    theme(axis_text_x=element_text(rotation=90, hjust=0.5)) +
    labs(
        title='VAF Distribution Across High-Density SNV Intervals',
        x='',
        y='VAF'
    ) +
    theme(figure_size=(25, 20)) +
    expand_limits(y=vaf_long_df['VAF'].max() + 0.01)
)

ggsavefig_and_show(plot, "vaf_distribution_high_snv_density_intervals")



# %%
# INFO: SNV-density-based Filtering!!

colotb_snvs_shared_final_filtered_pruned = colotb_snvs_shared_final_filtered[
    ~(colotb_snvs_shared_final_filtered["SNVid"].isin(density_filter_snvid))
    ].reset_index(drop=True)

colotb_snvs_shared_final_filtered_set = set(colotb_snvs_shared_final_filtered["SNVid"].values)

# INFO: Make VCFs for Total, each RE + None_RE
# INFO: Total SNVs
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"

pre_vcf = colotb_snvs_shared_final_filtered_pruned.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.pre.vcf",
    sep="\t",
    index=False,
)

# NOTE: Generate Pseudo VCF Header
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {outdir}/pseudovcf_header"
)

# NOTE: Combine Pseudo VCF Header with Pre VCF
os.system(
    f"cat {outdir}/pseudovcf_header \
          {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.pre.vcf \
          > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.vcf"
)

# NOTE:  Compress and Index VCF
os.system(
    f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.vcf \
          && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.vcf.gz"
)

# NOTE: Remove Pre VCF
os.system(
    f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.pre.vcf"
)

# INFO: For None_RE
retype = "None_RE"
pre_vcf = colotb_snvs_shared_final_filtered_pruned[
    (vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").isna())
    ].iloc[:, :10].rename(columns={"CHROM": "#CHROM"})

pre_vcf.to_csv(f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf", sep="\t", index=False)

os.system(f"cat {outdir}/pseudovcf_header \
          {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf \
            > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf")

# NOTE: Compress and Index VCF
os.system(
    f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf \
          && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf.gz"
)

# NOTE: Remove Pre VCF
os.system(
    f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf"
)

# NOTE: For REs

colotb_snvs_shared_final_filtered_pruned_re = colotb_snvs_shared_final_filtered_pruned[
    ~(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").isna())
    ].reset_index(drop=True)

for retype in ['DNA', 'LINE', 'LTR', 'Low_complexity', 'Retroposon', 'SINE', 'Satellite', 'Simple_repeat']:
      pre_vcf = colotb_snvs_shared_final_filtered_pruned_re[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_re, "RM").str.contains(retype)].iloc[:, :10]
      pre_vcf.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'COLO829T_PassageB_DSA']
      pre_vcf.to_csv(f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf", sep="\t", index=False)
      
      os.system(f"cat {outdir}/pseudovcf_header \
                {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf \
                  > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf")

      os.system(
          f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf \
                && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.vcf.gz")

      os.system(f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{retype}.pre.vcf")

# NOTE: Remove Pseudo VCF Header
os.system(
    f"rm {outdir}/pseudovcf_header"
)

# %%
# INFO: VAF distribution for SCNA-DSG-Density-Filtered SNVs

# NOTE: COLO829T Passage B Shared SNVs
colotb_snvs_shared_final_filtered_pruned_vaf = vcf_format_getter(
    colotb_snvs_shared_final_filtered_pruned, "VAF"
    )

n_snvs_colotb_snvs_shared_final_filtered_pruned = len(set(colotb_snvs_shared_final_filtered_pruned["SNVid"].values))

tmp_df = pd.DataFrame({'VAF': colotb_snvs_shared_final_filtered_pruned_vaf})

plot = (ggplot(tmp_df, aes(x='VAF')) +
        geom_histogram(breaks=np.linspace(0, 1, 151), fill='purple', alpha=0.7, size=0.1, color='black') +
        scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
        labs(title=f"Somatic SNVs Shared between Passage B and A (N={n_snvs_colotb_snvs_shared_final_filtered_pruned:,})") +
        theme_minimal() +
        theme(figure_size=(3, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black'),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=8, color='black', hjust=0),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, "colotb_snvs_shared_final_filtered_vaf")

# %%
# INFO: VAF distribution across SCNA states
colotb_snvs_shared_final_filtered_pruned_nonoverlapped = colotb_snvs_shared_final_filtered_pruned[~(colotb_snvs_shared_final_filtered_pruned["SNVid"].isin(overlapped_leftover_snv_list))].reset_index(drop=True)

colors_palette = ["#191970",
                  "#02ff00", 
                  "#ff6509",
                  "#00bfff"]

snv_cn1_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == "CN1")], "VAF")).reset_index(drop=True)
snv_cn1_vaf.columns = ["VAF"]
snv_cn1_vaf["CN"] = "CN1"
snv_cn2_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == "CN2")], "VAF")).reset_index(drop=True)
snv_cn2_vaf.columns = ["VAF"]
snv_cn2_vaf["CN"] = "CN2"
snv_cn3_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == "CN3")], "VAF")).reset_index(drop=True)
snv_cn3_vaf.columns = ["VAF"]
snv_cn3_vaf["CN"] = "CN3"
snv_cn4_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == "CN4")], "VAF")).reset_index(drop=True)
snv_cn4_vaf.columns = ["VAF"]
snv_cn4_vaf["CN"] = "CN4"
snv_cn5_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == ">CN4")], "VAF")).reset_index(drop=True)
snv_cn5_vaf.columns = ["VAF"]
snv_cn5_vaf["CN"] = ">CN4"

snv_cn1_4_vaf = pd.concat([snv_cn1_vaf, snv_cn2_vaf, snv_cn3_vaf, snv_cn4_vaf], axis=0)
snv_cn1_5_vaf = pd.concat([snv_cn1_vaf, snv_cn2_vaf, snv_cn3_vaf, snv_cn4_vaf, snv_cn5_vaf], axis=0)

plot = (
    ggplot(snv_cn1_4_vaf, aes(x='CN', y='VAF', color='CN')) +
    geom_point(position=position_jitter(width=0.25, height=0), 
               alpha=0.6, size=1.5) +
    scale_color_manual(values=colors_palette) +
    theme_minimal() +
    theme(axis_text_x=element_text(rotation=90, hjust=0.5)) +
    labs(
        title='',
        x='Copy Number',
        y='VAF'
    ) +
    theme(figure_size=(25, 20),
          text=element_text(family='Arial'),
          axis_text_x=element_text(rotation=0, hjust=0.5, size=24, color='black'),
          axis_text_y=element_text(rotation=0, hjust=0.5, size=24, color='black'),
          axis_title=element_text(size=30, color='black'),
          plot_title=element_text(size=30, color='black'))
)

ggsavefig_and_show(plot, "vaf_distribution_across_scna")

# %%
plot = (
    ggplot(snv_cn1_4_vaf, aes(x='VAF', fill='CN')) +
    geom_histogram(bins=50, alpha=0.7) +
    scale_x_continuous(limits=[0, 1]) +
    scale_y_log10() +
    facet_wrap('~CN', ncol=2, scales='free_y') +
    scale_fill_manual(values=colors_palette) +
    theme_minimal() +
    labs(
        title='VAF Distribution by Copy Number',
        x='VAF',
        y='Count'
    ) +
    theme(figure_size=(12, 8),
          strip_text=element_text(size=12, color='black'),
          axis_text=element_text(size=10, color='black'),
          axis_title=element_text(size=12, color='black'),
          plot_title=element_text(size=14, color='black'))
)
ggsavefig_and_show(plot, "vaf_histograms_by_cn")

# %%
plot = (
    ggplot(snv_cn1_4_vaf, aes(x='VAF', fill='CN')) +
    geom_histogram(bins=50, alpha=0.7) +
    scale_x_continuous(limits=[0, 1]) +
    facet_wrap('~CN', ncol=2, scales='free_y') +
    scale_fill_manual(values=colors_palette) +
    theme_minimal() +
    labs(
        title='VAF Distribution by Copy Number',
        x='VAF',
        y='Count'
    ) +
    theme(figure_size=(12, 8),
          strip_text=element_text(size=12, color='black'),
          axis_text=element_text(size=10, color='black'),
          axis_title=element_text(size=12, color='black'),
          plot_title=element_text(size=14, color='black'))
)
ggsavefig_and_show(plot, "vaf_histograms_by_cn_nonlog")


# %%
y_limits = {
    'CN1': [0, 10000],
    'CN2': [0, 25000],
    'CN3': [0, 1750],
    'CN4': [0, 700]
}

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
axes = [ax1, ax2, ax3, ax4]
fig.suptitle('VAF Distribution by Copy Number', fontsize=14)

for ax in axes:
    ax.clear()

for idx, (cn, ax) in enumerate(zip(['CN1', 'CN2', 'CN3', 'CN4'], axes)):
    data_subset = snv_cn1_4_vaf[snv_cn1_4_vaf['CN'] == cn]

    counts, bins, patches = ax.hist(data_subset['VAF'], bins=30, alpha=0.7,
                                    color=colors_palette[idx], edgecolor='black', linewidth=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(y_limits[cn])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.set_xlabel('VAF', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(cn, fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()

savefig_and_show("vaf_histograms_by_cn_nonlog")

# %%
y_limits = {
    'CN1': [0, 10000],
    'CN2': [0, 25000],
    'CN3': [0, 1750],
    'CN4': [0, 700]
}

cn_categories = ['CN1', 'CN2', 'CN3', 'CN4']

for idx, cn in enumerate(cn_categories):
    data_subset = snv_cn1_4_vaf[snv_cn1_4_vaf['CN'] == cn].copy()

    plot = (ggplot(data_subset, aes(x='VAF')) +
            geom_histogram(breaks=np.linspace(0, 1, 31), fill=colors_palette[idx], alpha=0.7, 
                          size=0.5, color='black') +
            scale_x_continuous(limits=[0, 1]) +
            scale_y_continuous(limits=y_limits[cn], expand=(0, 0, 0.05, 0), 
                              labels=comma_format()) +
            labs(title=f"{cn}", 
                 x='VAF', 
                 y='Count') +
            theme_minimal() +
            theme(figure_size=(3, 2.5),
                  text=element_text(family='Arial'),
                  axis_text_x=element_text(color='black'),
                  axis_text_y=element_text(color='black'),
                  plot_title=element_text(size=12, color='black', hjust=0.5),
                  axis_line_x=element_line(size=0.5, color='black'),
                  axis_line_y=element_line(size=0.5, color='black'),
                  axis_ticks_major=element_line(size=0.5, color='black'),
                  axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
                  axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
            )

    ggsavefig_and_show(plot, f"vaf_histogram_{cn.lower()}")

# %%
# INFO: Check CN4 SNVs
colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4 = colotb_snvs_shared_final_filtered_pruned_nonoverlapped[
    vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped, "CN") == "CN4"].reset_index(drop=True)

colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4_chr1 = colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4[colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4["CHROM"] == 'haplotype1-0000012'].reset_index(drop=True)

colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4_chr7 = colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4[colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4["CHROM"] == 'haplotype1-0000009'].reset_index(drop=True)


snv_cn4_chr1_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4_chr1, "VAF")).reset_index(drop=True)
snv_cn4_chr1_vaf.columns = ["VAF"]
snv_cn4_chr1_vaf["chromosome"] = "chr1"

snv_cn4_chr7_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned_nonoverlapped_cn4_chr7, "VAF")).reset_index(drop=True)
snv_cn4_chr7_vaf.columns = ["VAF"]
snv_cn4_chr7_vaf["chromosome"] = "chr7"

snv_cn4_vaf = pd.concat([snv_cn4_chr1_vaf, snv_cn4_chr7_vaf], axis=0)

fig,ax = plt.subplots(1,1, figsize=(6,4), constrained_layout=True)
sns.histplot(data=snv_cn4_vaf, hue="chromosome", x="VAF", bins=30, stat="probability", common_norm=False, alpha=0.5, ax=ax)
ax.set_xlim(0,1)


# %%
# INFO: VAF by Repeat Elements
repeats_of_interest = ['LINE', 'Satellite', 'SINE', 'LTR', 'DNA', 'Simple_repeat', 'Retroposon', 'Low_complexity']

colors_palette = {'Satellite': '#ff0035',
                  'Simple_repeat': '#ffbe0b',
                  'LINE': '#d6ff23',
                  'SINE': '#65d100',
                  'LTR': '#00fbb6',
                  'DNA': '#071f35',
                  'Retroposon': '#446d92',
                  'Low_complexity': '#98b1c8',
                  'None_RE': '#c75cf5'}

colors_palette_repeat = list(map(lambda x: colors_palette[x], repeats_of_interest))

# NOTE: REs
for idx, repeat_name in enumerate(repeats_of_interest):
    #data_subset = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, repeatmasker_pr[repeatmasker_pr.Name1 == repeat_name].merge()) # NOTE: Way Too slow
    data_subset = colotb_snvs_shared_final_filtered_pruned[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").str.contains(repeat_name)) & (~vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").isna())]
    data_subset_vaf = pd.DataFrame(vcf_format_getter(data_subset, "VAF"))
    data_subset_vaf.columns = ["VAF"]
    plot = (ggplot(data_subset_vaf, aes(x='VAF')) +
            geom_histogram(breaks=np.linspace(0, 1, 31), fill=colors_palette_repeat[idx], alpha=0.7, 
                          size=0.5, color='black') +
            scale_x_continuous(limits=[0, 1]) +
            scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
            labs(title=f"{repeat_name}", 
                 x='VAF', 
                 y='Count') +
            theme_minimal() +
            theme(figure_size=(3, 2.5),
                  text=element_text(family='Arial'),
                  axis_text_x=element_text(color='black'),
                  axis_text_y=element_text(color='black'),
                  plot_title=element_text(size=12, color='black', hjust=0.5),
                  axis_line_x=element_line(size=0.5, color='black'),
                  axis_line_y=element_line(size=0.5, color='black'),
                  axis_ticks_major=element_line(size=0.5, color='black'),
                  axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
                  axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
            )

    ggsavefig_and_show(plot, f"vaf_histogram_{repeat_name}")

# NOTE: None-RE
#data_subset = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, callable_pr.subtract(repeatmasker_pr))
data_subset = colotb_snvs_shared_final_filtered_pruned[(vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").isna())]
data_subset_vaf = pd.DataFrame(vcf_format_getter(data_subset, "VAF"))
data_subset_vaf.columns = ["VAF"]

plot = (ggplot(data_subset_vaf, aes(x='VAF')) +
        geom_histogram(breaks=np.linspace(0, 1, 31), fill=colors_palette['None_RE'], alpha=0.7, 
                      size=0.5, color='black') +
        scale_x_continuous(limits=[0, 1]) +
        scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
        labs(title=f"No Repeat Elements", 
             x='VAF', 
             y='Count') +
        theme_minimal() +
        theme(figure_size=(3, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black'),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.5, color='black'),
              axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
        )

ggsavefig_and_show(plot, f"vaf_histogram_None_RE")

# %%
y_limits = {
    'CN1': [0, 10000],
    'CN2': [0, 25000],
    'CN3': [0, 1750],
    'CN4': [0, 700]
}

cn_categories = ['CN1', 'CN2', 'CN3', 'CN4']

colors_palette = ["#191970",
                  "#02ff00", 
                  "#ff6509",
                  "#00bfff"] # NOTE: Need to redefine

for idx, cn in enumerate(cn_categories):
    data_subset = snv_cn1_4_vaf[snv_cn1_4_vaf['CN'] == cn].copy()

    plot = (ggplot(data_subset, aes(x='VAF')) +
            geom_histogram(breaks=np.linspace(0, 1, 31), fill=colors_palette[idx], alpha=0.7, 
                          size=0.5, color='black') +
            scale_x_continuous(limits=[0, 1]) +
            scale_y_continuous(limits=y_limits[cn], expand=(0, 0, 0.05, 0), 
                              labels=comma_format()) +
            labs(title=f"{cn}", 
                 x='VAF', 
                 y='Count') +
            theme_minimal() +
            theme(figure_size=(3, 2.5),
                  text=element_text(family='Arial'),
                  axis_text_x=element_text(color='black'),
                  axis_text_y=element_text(color='black'),
                  plot_title=element_text(size=12, color='black', hjust=0.5),
                  axis_line_x=element_line(size=0.5, color='black'),
                  axis_line_y=element_line(size=0.5, color='black'),
                  axis_ticks_major=element_line(size=0.5, color='black'),
                  axis_ticks_minor_x=element_line(size=0.5, color='darkgray'),
                  axis_ticks_minor_y=element_line(size=0.5, color='darkgray'))
            )

    ggsavefig_and_show(plot, f"vaf_histogram_{cn.lower()}")

# %%
colo829_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']

chrom_color_palette = sns.hls_palette(n_colors=len(colo829_chromosomes), l=.3)
chrom_color_palette = {key: rgb_to_hex(color) for key, color in zip(colo829_chromosomes, chrom_color_palette)}

all_chr_data = []

for chrom in colo829_chromosomes:
    contig_chr = df_paf[df_paf['primary_chromosome'] == chrom]['query_name'].values
    contig_chr_vcf = colotb_snvs_shared_final_filtered_pruned_nonoverlapped[colotb_snvs_shared_final_filtered_pruned_nonoverlapped['CHROM'].isin(contig_chr)]
    
    all_cn_values = vcf_info_getter(contig_chr_vcf, "CN")
    all_cn_values = all_cn_values.dropna()
    
    valid_cn_mask = ~all_cn_values.astype(str).str.startswith('>')
    unique_cns = all_cn_values[valid_cn_mask].unique()
    
    for cn in unique_cns:
        cn_mask = all_cn_values == cn
        contig_chr_vcf_cn = contig_chr_vcf[cn_mask]
        
        contig_chr_vaf = pd.DataFrame(vcf_format_getter(contig_chr_vcf_cn, "VAF")).reset_index(drop=True)
        contig_chr_vaf.columns = ["VAF"]
        contig_chr_vaf["chromosome"] = chrom
        contig_chr_vaf["CN"] = f"{cn}"
        
#        # Only add if there are at least 10 SNVs
#        if contig_chr_vaf.shape[0] >= 10:
#            all_chr_data.append(contig_chr_vaf)

        all_chr_data.append(contig_chr_vaf)

combined_data = pd.concat(all_chr_data, ignore_index=True)

combined_data['chromosome'] = pd.Categorical(
    combined_data['chromosome'], 
    categories=colo829_chromosomes,
    ordered=True
)

combined_data['chrom_cn'] = combined_data['chromosome'].astype(str) + ' h' + combined_data['CN']

chrom_cn_order = []
for chrom in colo829_chromosomes:
    chrom_data = combined_data[combined_data['chromosome'] == chrom]
    if not chrom_data.empty:
        def extract_cn_number(cn_str):
            try:
                if cn_str.startswith('CN'):
                    return int(cn_str[2:])
                else:
                    return int(cn_str)
            except:
                return 0
        
        cn_values = sorted(chrom_data['CN'].unique(), key=extract_cn_number)
        for cn in cn_values:
            chrom_cn_order.append(f"{chrom} h{cn}")

combined_data['chrom_cn'] = pd.Categorical(
    combined_data['chrom_cn'],
    categories=chrom_cn_order,
    ordered=True
)

annotation_data = []
for chrom in combined_data['chromosome'].unique():
    for cn in combined_data['CN'].unique():
        subset = combined_data[(combined_data['chromosome'] == chrom) & (combined_data['CN'] == cn)]
        if not subset.empty:
            n_snvs = len(subset)
            annotation_data.append({
                'chromosome': chrom,
                'CN': cn,
                'chrom_cn': f"{chrom} h{cn}",
                'x': 0.95,
                'y': float('inf'),
                'label': f'N={n_snvs}'
            })

annotation_df = pd.DataFrame(annotation_data)

annotation_df['chromosome'] = pd.Categorical(
    annotation_df['chromosome'], 
    categories=colo829_chromosomes,
    ordered=True
)

plot = (ggplot(combined_data, aes(x='VAF', fill='chrom_cn')) +
        geom_histogram(breaks=np.linspace(0, 1, 31), alpha=0.7, size=0.5, color='black') +
        geom_text(data=annotation_df, 
                 mapping=aes(x='x', y='y', label='label'), 
                 inherit_aes=False, 
                 ha='right', va='top', 
                 size=8, color='black',
                 nudge_x=0, nudge_y=0) +
        facet_grid('chromosome ~ CN', scales='free_y') +
        scale_fill_manual(values={combo: chrom_color_palette[combo.split(' ')[0]] 
                                 for combo in chrom_cn_order}) +
        scale_x_continuous(limits=[0, 1]) +
        scale_y_continuous(expand=(0.05, 0, 0.15, 0)) +
        labs(x='VAF', y='Count') +
        theme_minimal() +
        theme(figure_size=(12, 20),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', size=6),
              axis_text_y=element_text(color='black', size=6),
              strip_text=element_text(size=8, color='black'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              legend_position='none')
        )

ggsavefig_and_show(plot, "vaf_histogram_colo829_all_chromosomes_cn")


# %%
combined_data_breakdown = combined_data.groupby('chromosome').agg({
    'VAF': ['count', lambda x: (x < 0.8).sum(), lambda x: (x < 0.8).mean()]
}).round(4)

combined_data_breakdown.columns = ['total_count', 'VAF_below_0.8_count', 'proportion_VAF_below_0.8']

fig, ax = plt.subplots(1,1, figsize=(2,5))
sns.swarmplot(data=combined_data_breakdown, 
              y='proportion_VAF_below_0.8',
              size=8,
              alpha=0.8,
              color='coral',
              ax=ax)
ax.set_ylabel("Proportion of SNVs with VAF<0.8")
savefig_and_show("proportion_of_snvs_with_vaf_below_0.8_by_chromosomes")

# %% [markdown]
# INFO: Double-base substitution
# %%
def find_consecutive_snvs(df: pd.DataFrame, consecutive: int = 2):
    df_sorted = df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)

    df_sorted["pos_diff"] = df_sorted.groupby("CHROM")["POS"].diff()
    df_sorted["group_id"] = (
        (df_sorted["pos_diff"] != 1) | df_sorted["pos_diff"].isna()
    ).cumsum()

    group_sizes = df_sorted.groupby(["CHROM", "group_id"]).size()

    consecutive_groups = group_sizes[group_sizes == consecutive].index

    consecutive_mask = df_sorted.set_index(["CHROM", "group_id"]).index.isin(
        consecutive_groups
    )

    return (
        df_sorted[consecutive_mask]
        .drop(["pos_diff", "group_id"], axis=1)
        .reset_index(drop=True)
    )


# %%
colotb_dbs_shared_final = (
    pd.concat(
        [
            find_consecutive_snvs(colotb_snvs_shared_final_filtered_pruned, consecutive=2),
            find_consecutive_snvs(colotb_snvs_shared_final_filtered_pruned, consecutive=4),
        ],
        axis=0,
    )
    .sort_values(by=["CHROM", "POS"])
    .reset_index(drop=True)
)


# INFO: COLO829T Passage B
# INFO: Shared DBS Variants
pre_vcf = colotb_dbs_shared_final.iloc[:, :10].rename(columns={"CHROM": "#CHROM"})
pre_vcf.to_csv(
    f"{outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.pre.vcf",
    sep="\t",
    index=False,
)

# NOTE: Generate Pseudo VCF Header
os.system(
    f"zcat '{dir}/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz' \
          | awk '/^##/' \
          > {outdir}/pseudovcf_header"
)

os.system(
    f"cat {outdir}/pseudovcf_header \
          {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.pre.vcf \
          > {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.vcf"
)

os.system(
    f"bgzip -f {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.vcf \
          && tabix -p vcf {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.vcf.gz"
)

os.system(
    f"rm {outdir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dbs.pre.vcf"
)
os.system(
    f"rm {outdir}/pseudovcf_header"
)

# %%
# NOTE: whatshap phase to phase 0/1 DBSs (1/1 variants don't need to be phased)

# %%
"""
colotb_snvs_shared_final_nearest_distance = pd.concat(
    [
        colotb_snvs_shared_final.groupby("CHROM")["POS"].diff(),
        colotb_snvs_shared_final.groupby("CHROM")["POS"].diff().shift(-1),
    ],
    axis=1, 
).min(axis=1)

fig, axes = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
sns.histplot(
    np.log10(colotb_snvs_shared_final_nearest_distance), color="#4B2E83", ax=axes
)
axes.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)
axes.set_xlabel("Distance to the Nearest SNVs (Log 10)")
"""
# %%

"""
new_cols = [
    "chrom",
    "start",
    "end",
    "pgfbSNVid",
    "HapOfOrigin",
    "strand",
    "segmentchrom",
    "segmentStart",
    "segmentEnd",
    "CN",
    "Overlap",
]

colotb_snvs_final_pgfb_with_cna = pl.read_csv(
    f"{cnadir}/Intersect_wo_COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot_peaks__surj_onto_COLO829BL_hap1-2_withTags_WITH_merged_cov_callable_100kb_log2ratio_CBS_wCN_simple.bed.gz",
    separator="\t",
    has_header=False,
    new_columns=new_cols
)

"""

# %%
# INFO: Evidence across different platforms by coverage threshold
colotb_snvs_shared_final_filtered_pruned_set = set(colotb_snvs_shared_final_filtered_pruned["SNVid"].values)

platforms = {
    "ONT": colotba_snv_ont_newmotor_pileup_vcf,
    "UL-ONT": colotba_snv_ulont_pileup_vcf,
    "ONT_MAPQ1": colotba_snv_ont_newmotor_mapq1_pileup_vcf,
    "UL-ONT_MAPQ1": colotba_snv_ulont_mapq1_pileup_vcf,
    "Illumina": colotba_snv_illumina_pileup_vcf,
    "Element": colotba_snv_element_pileup_vcf,
}

evidence_platform_by_coverage = dict()
for platform_key in platforms:
    evidence_platform_by_coverage[platform_key] = list()  # cov1, cov2... cov5, cov10

for platform_key, platform in platforms.items():
    print(platform_key)
    for i in range(1, 6):  # cov1 through cov5
        evidence_platform_by_coverage[platform_key].append(
            platform[vcf_format_getter(platform, "AD") >= i]["SNVid"]
            .isin(colotb_snvs_shared_final_filtered_pruned_set)
            .value_counts()[True]
        )

    # NOTE: For cov10
    evidence_platform_by_coverage[platform_key].append(
        platform[vcf_format_getter(platform, "AD") >= 10]["SNVid"]
        .isin(colotb_snvs_shared_final_filtered_pruned_set)
        .value_counts()[True]
    )

evidence_platform_by_coverage_df = pd.DataFrame(
    evidence_platform_by_coverage,
    index=["Cov1", "Cov2", "Cov3", "Cov4", "Cov5", "Cov10"],
)

evidence_platform_by_coverage_df_percentage = (
    evidence_platform_by_coverage_df * 100 / len(colotb_snvs_shared_final_filtered_pruned_set)
)

evidence_platform_by_coverage_df_percentage_melted = (
    evidence_platform_by_coverage_df_percentage.reset_index().melt(
        id_vars="index", var_name="Platform", value_name="Percentage"
    )
)

# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
sns.barplot(
    data=evidence_platform_by_coverage_df_percentage_melted,
    x="index",
    y="Percentage",
    hue="Platform",
    palette="husl",
    ax=ax,
)
ax.set_xlabel("Coverage")
ax.set_ylabel("Percentage")
ax.tick_params(axis="x", rotation=0)

ax.legend(
    title="Platform",
    loc="upper right",
    bbox_to_anchor=(0.99, 1.10),
    frameon=True,
    fancybox=False,
    title_fontsize=5,
    fontsize=4.5,
    edgecolor="k",
)
ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

savefig_and_show("evidence_platform_by_coverage")

# %%
# INFO: DSA-based Reference SNV-set injected to DSG and surjected to GRCh38
snv_referenceset_pgfbsnvid_snvid_tab = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dict.tsv", names=['pgfbSNVid', 'SNVid'])

pgfbsnvid_to_snvid = dict(zip(snv_referenceset_pgfbsnvid_snvid_tab['pgfbSNVid'], snv_referenceset_pgfbsnvid_snvid_tab['SNVid']))
snvid_to_pgfbsnvid= dict(zip(snv_referenceset_pgfbsnvid_snvid_tab['SNVid'], snv_referenceset_pgfbsnvid_snvid_tab['pgfbSNVid']))

snv_referenceset_hg38_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toGRCh38/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_GRCh38_withTags_sorted.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])

snv_referenceset_hg38_position["SNVid"] = snv_referenceset_hg38_position['pgfbSNVid'].map(pgfbsnvid_to_snvid)

hg38_primary_contigs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

colotb_snvs_shared_final_filtered_pruned_hg38_surjectable_primary_set = set(
    snv_referenceset_hg38_position[
        snv_referenceset_hg38_position['Chromosome'].isin(hg38_primary_contigs)
        ]["SNVid"].values
    )

# %%
sns.set_theme(font="Arial", font_scale=1.0, style="ticks")
matplotlib.rcParams["figure.dpi"] = 200
plt.rc("axes.spines", top=False, right=False)

total = len(
    (
        colotb_snvs_shared_final_filtered_pruned_set.union(
            colotb_snvs_shared_final_filtered_pruned_hg38_surjectable_primary_set
            )
    )
)
venn = venn2(
    [
        colotb_snvs_shared_final_filtered_pruned_set,
        colotb_snvs_shared_final_filtered_pruned_hg38_surjectable_primary_set,
    ],
    ("COLO829BL DSA", "GRCh38"),
    subset_label_formatter=lambda x: f"{x:,}\n({(x / total):.2%})",
)
venn.get_patch_by_id("10").set_color("#FFE3B3")
venn.get_patch_by_id("01").set_color("#53D2DC")
venn.get_patch_by_id("11").set_color("#4F8FC0")

for i in ["10", "01", "11"]:
    venn.get_patch_by_id(i).set_edgecolor("black")
    venn.get_patch_by_id(i).set_alpha(0.5)

sns.set_theme(font="Arial", font_scale=1.15, style="ticks")
matplotlib.rcParams["figure.dpi"] = 300
plt.rc("axes.spines", top=False, right=False)

savefig_and_show("venn_colotb_colota_snv_unique_to_each")

# %%
colotb_snvs_shared_final_filtered_pruned_dsa_only = colotb_snvs_shared_final_filtered_pruned[~(colotb_snvs_shared_final_filtered_pruned["SNVid"].isin(colotb_snvs_shared_final_filtered_pruned_hg38_surjectable_primary_set))].reset_index(drop=True)

repeat_category = ["None_RE",
                   "DNA",
                   "DNA?",
                   "LINE",
                   "LTR",
                   "Low_complexity",
                   "RC",
                   "Retroposon",
                   "SINE",
                   "Satellite",
                   "Simple_repeat",
                   "Unknown",
                   "rRNA",
                   "scRNA",
                   "snRNA",
                   "srpRNA",
                   "tRNA"] 

rc_snv_nonhg38_count = dict()
for rc in repeat_category[1:]:
    if rc == "DNA?":
       pattern = r"DNA\?"
    else:
        pattern = r"{}".format(rc)

    count = colotb_snvs_shared_final_filtered_pruned_dsa_only[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_dsa_only, "RM").str.contains(pattern, na=False)].shape[0]
    rc_snv_nonhg38_count[rc] = count

rc_snv_nonhg38_count["None_RE"] = colotb_snvs_shared_final_filtered_pruned_dsa_only[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned_dsa_only, "RM").isna()].shape[0]

rc_snv_nonhg38_count_df = pd.DataFrame.from_dict(rc_snv_nonhg38_count, orient='index', columns=["SNV_Count"])

rc_snv_nonhg38_count_df_new = rc_snv_nonhg38_count_df.copy()

mask = rc_snv_nonhg38_count_df_new['SNV_Count'] < 100
others_sum = rc_snv_nonhg38_count_df_new.loc[mask, 'SNV_Count'].sum()

rc_snv_nonhg38_count_df_new = rc_snv_nonhg38_count_df_new.loc[~mask]

rc_snv_nonhg38_count_df_new.loc['Others'] = others_sum

rc_snv_nonhg38_count_df_new_fraction = rc_snv_nonhg38_count_df_new*100 / rc_snv_nonhg38_count_df_new.sum()

# %%
rc_snv_nonhg38_color_palette = sns.color_palette(palette='gist_ncar_r', n_colors=rc_snv_nonhg38_count_df_new_fraction.shape[0])
rc_snv_nonhg38_color_map = {key: color for key, color in zip(rc_snv_nonhg38_count_df_new_fraction.index.values, rc_snv_nonhg38_color_palette)}
rc_snv_nonhg38_color_map_hex = {category: rgb_to_hex(color) for category, color in rc_snv_nonhg38_color_map.items()}

specified_order = ["None_RE", "Satellite", "Simple_repeat", "LINE", "SINE", "LTR", "Others"]
rc_snv_nonhg38_count_df_new_fraction = rc_snv_nonhg38_count_df_new_fraction.loc[specified_order]

rc_snv_nonhg38_count_df_new_fraction_plot = rc_snv_nonhg38_count_df_new_fraction.reset_index()
rc_snv_nonhg38_count_df_new_fraction_plot.columns = ['Category', 'Percentage']

rc_snv_nonhg38_count_df_new_fraction_plot['Group'] = 'SNV_Distribution'

rc_snv_nonhg38_count_df_new_fraction_plot['Category'] = pd.Categorical(
    rc_snv_nonhg38_count_df_new_fraction_plot['Category'], 
    categories=specified_order, 
    ordered=True
    )

plot = (ggplot(rc_snv_nonhg38_count_df_new_fraction_plot, aes(x='Group', y='Percentage', fill='Category')) +
        geom_col(width=0.3) +
        labs(title='RE category for DSA-specific SNVs (N={:,})'.format(colotb_snvs_shared_final_filtered_pruned_dsa_only.shape[0]),
             x='',
             y='Percentage (%)',
             fill='Repeat Element') +
        theme_tufte() +
        theme(axis_text_x=element_blank(),
              axis_ticks_major_x=element_blank(),
              figure_size=(4, 3)) +
        scale_fill_manual(values=rc_snv_nonhg38_color_map_hex) +
        scale_y_continuous(limits=[0, 100]) + 
        guides(fill=guide_legend(reverse=False)) +
        theme(
            text=element_text(family='Arial'),
            axis_text_y=element_text(color='black'),
            plot_title=element_text(color='black', size=7)
        )
)

print(plot)

ggsavefig_and_show(plot, "rc_snv_nonhg38_count_distribution")

# %%
# INFO: Benchmarking Flagship Paper SNVs Flagged sites
snv_flagset_outdir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV"

snv_flagset = snv_flagset_plus_referenceset - colotb_snvs_shared_final_filtered_pruned_set

snv_record_from_bl = colobl_snvs_correct[colobl_snvs_correct["SNVid"].isin(snv_flagset)]
snv_record_from_bl.rename(columns={"COLO829BL": "Flagged"}, inplace=True)

snv_record_from_tb = colotb_snvs_correct[colotb_snvs_correct["SNVid"].isin(snv_flagset)]
snv_record_from_tb.rename(columns={"COLO829T_PassageB_DSA": "Flagged"}, inplace=True)

snv_record_from_ta = colota_snvs_correct[colota_snvs_correct["SNVid"].isin(snv_flagset)]
snv_record_from_ta.rename(columns={"COLO829T_PassageA_DSA": "Flagged"}, inplace=True)

snv_flagset_vcf = pd.concat([snv_record_from_bl, snv_record_from_tb, snv_record_from_ta], axis=0).reset_index(drop=True)

snv_flagset_position = snv_flagset_vcf[["CHROM", "POS"]].copy().drop_duplicates().reset_index(drop=True)
snv_flagset_position = pd.concat([snv_flagset_position["CHROM"], snv_flagset_position["POS"]-1, snv_flagset_position["POS"]], axis=1)

snv_flagset_position['new1'] = [f'SNV_{i+1}' for i in range(len(snv_flagset_position))]
snv_flagset_position['new2'] = 1 # NOTE: Placeholder
snv_flagset_position['new3'] = '+'

snv_flagset_outdir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV"
snv_flagset_position.to_csv(f"{snv_flagset_outdir}/SNV_Flagset.bed", header=None, index=None, sep="\t")


# INFO: Mutational Spectrum Analysis using Somatic SNV accounting for K-mer Frequency across different contexts

# INFO: First Calculate K-mer Frequency across the DSA
# %%
kmer3_file="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/DSA_COLO829BL_v3.0.0_Flagger-NucFlag_100kb-DEL_removed.fasta_3mers.txt"
kmer3_canonical_file="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/DSA_COLO829BL_v3.0.0_Flagger-NucFlag_100kb-DEL_removed.fasta_3mers_canonical.txt"

kmer3_dict = dict()
kmer3_canonical_dict = dict()

with open(kmer3_file, 'r') as dfh:
    c = 1
    for line in dfh:
        if c % 2 == 1:
            kmer_count = int(line.strip().lstrip(">"))
        elif c % 2 == 0:
            kmer3_dict[line.strip()] = kmer_count
        c += 1

with open(kmer3_canonical_file, 'r') as dfh:
    c = 1
    for line in dfh:
        if c % 2 == 1:
            kmer_count = int(line.strip().lstrip(">"))
        elif c % 2 == 0:
            kmer3_canonical_dict[line.strip()] = kmer_count
        c += 1

kmer3_canonical_sbs_dict = dict()
for kmer, count in kmer3_canonical_dict.items():
    if kmer[1] not in ["C", "T"]:
        kmer3_canonical_sbs_dict[reverse_complement(kmer)] = count
    else:
        kmer3_canonical_sbs_dict[kmer] = count

kmer3_canonical_sbs_dict = dict(sorted(kmer3_canonical_sbs_dict.items()))



# %%
df_canonical_sbs96 = pd.DataFrame(list(kmer3_canonical_sbs_dict.items()), columns=['3mer', 'count'])
df_canonical_sbs96["count_log10"] = np.log10(df_canonical_sbs96['count'])

plot = (ggplot(df_canonical_sbs96, aes(x='3mer', y='count')) +
        geom_col(fill='steelblue', alpha=0.8) +
        theme_minimal() +
        labs(title='Canonical 3-mers in the DSA in SBS96 trinucleotides context',
             x='3-mer',
             y='Count') +
        scale_y_continuous(labels=scientific_format(digits=1)) +
        theme(axis_text_x=element_text(rotation=45, hjust=0.5),
              plot_title=element_text(size=12)))

ggsavefig_and_show(plot, "kmer3_canonical_sbs96_sbs_count")

plot = (ggplot(df_canonical_sbs96, aes(x='3mer', y='count_log10')) +
        geom_col(fill='steelblue', alpha=0.8) +
        theme_minimal() +
        labs(title='Canonical 3-mers in the DSA in SBS96 trinucleotides context',
             x='3-mer',
             y='Count (log10)') +
        theme(axis_text_x=element_text(rotation=45, hjust=0.5),
              plot_title=element_text(size=12)))

ggsavefig_and_show(plot, "kmer3_canonical_sbs96_sbs_count_log10")

"""
df = pd.DataFrame(list(kmer3_dict.items()), columns=['3mer', 'count'])
df["count_log10"] = np.log10(df['count'])

plot = (ggplot(df, aes(x='3mer', y='count')) +
        geom_col(fill='steelblue', alpha=0.8) +
        theme_minimal() +
        labs(title='Bar Plot of Non-Canonical 3-mers in the DSA',
             x='3-mer',
             y='Count') +
        scale_y_continuous(labels=scientific_format(digits=1)) +
        theme(axis_text_x=element_text(rotation=45, hjust=0.5, size=7)))

ggsavefig_and_show(plot, "kmer3_noncanonical_sbs_count")

plot = (ggplot(df, aes(x='3mer', y='count_log10')) +
        geom_col(fill='steelblue', alpha=0.8) +
        theme_minimal() +
        labs(title='Bar Plot of Non-Canonical 3-mers in the DSA',
             x='3-mer',
             y='Count (log10)') +
        theme(axis_text_x=element_text(rotation=45, hjust=0.5, size=7)))

ggsavefig_and_show(plot, "kmer3_noncanonical_sbs_count_log10")
"""

# %%
df_canonical_sbs96 = df_canonical_sbs96[["3mer", "count"]].set_index("3mer")
df_canonical_sbs96_fraction = df_canonical_sbs96.div(df_canonical_sbs96.sum(axis=0), axis=1)

df_canonical_sbs96_fraction.to_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/RepeatMasker/kmer3_canonical_sbs_fraction.tsv", sep="\t")
# %%
kmer_repeat_filelist = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/RepeatMasker/*3mers_canonical.txt"))
repeat_category = ["None_RE",
                   "DNA",
                   "DNA?",
                   "LINE",
                   "LTR",
                   "Low_complexity",
                   "RC",
                   "Retroposon",
                   "SINE",
                   "Satellite",
                   "Simple_repeat",
                   "Unknown",
                   "rRNA",
                   "scRNA",
                   "snRNA",
                   "srpRNA",
                   "tRNA"] # NOTE: Redundant (defined above but leave it here for clarity)

# INFO: Checking for relevant category (checking RE category with ≥100 SNVs)
#for i in repeat_category[1:]:
#    if i == "DNA?":
#        pattern = r"DNA\?"
#    else:
#        pattern = i
#    
#    count = colotb_snvs_shared_final_filtered_pruned[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").str.contains(pattern, #na=False)].shape[0]
#    print(i, count)


kmer_repeat_filelist_dict = dict(zip(repeat_category, kmer_repeat_filelist))
kmer3_canonical_sbs_df_combined = dict()

for rc, kmer_repeat_file in kmer_repeat_filelist_dict.items():
    kmer3_canonical_dict = dict()
    with open(kmer_repeat_file, 'r') as dfh:
        c = 1
        for line in dfh:
            if c % 2 == 1:
                kmer_count = int(line.strip().lstrip(">"))
            elif c % 2 == 0:
                kmer3_canonical_dict[line.strip()] = kmer_count
            c += 1
            
    kmer3_canonical_sbs_dict = dict()
    for kmer, count in kmer3_canonical_dict.items():
        if kmer[1] not in ["C", "T"]:
            kmer3_canonical_sbs_dict[reverse_complement(kmer)] = count
        else:
            kmer3_canonical_sbs_dict[kmer] = count

    kmer3_canonical_sbs_dict = dict(sorted(kmer3_canonical_sbs_dict.items()))

    kmer3_canonical_sbs_df = pd.DataFrame([kmer3_canonical_sbs_dict]).T
    kmer3_canonical_sbs_df.index.name = "3mer"
    kmer3_canonical_sbs_df.columns = ["count"]

    kmer3_canonical_sbs_df_combined[rc] = kmer3_canonical_sbs_df


combined_df = pd.concat(kmer3_canonical_sbs_df_combined.values(), axis=1, keys=kmer3_canonical_sbs_df_combined.keys())
combined_df.columns = combined_df.columns.get_level_values(0)

combined_df_fraction = combined_df.div(combined_df.sum(axis=0), axis=1)

combined_df_fraction_melted = combined_df_fraction.reset_index().melt(id_vars='3mer', var_name='RE', value_name='Fraction')

colors_palette = ["#15ffa9",
                  "#0000ff",
                  "#ff0b03",
                  "#280021",
                  "#f2ff02",
                  "#1b9cff",
                  "#fffbc6",
                  "#fe007f",
                  "#fc1df8",
                  "#00584c",
                  "#ffb2e1",
                  "#7b2b00",
                  "#ffa200",
                  "#5b8900",
                  "#00eaff",
                  "#00ff00",
                  "#021d94"]

plot = (ggplot(combined_df_fraction_melted, aes(x='3mer', y='Fraction', color='RE', group='RE')) +
        geom_line() +
        geom_point() +
        theme_minimal() +
        theme(
            legend_position='right',
            figure_size=(12, 8),
            text=element_text(family='Arial'),
            axis_text_x=element_text(color='black', angle=45, hjust=0.5),
            axis_text_y=element_text(color='black')
            ) +
        scale_color_manual(values=colors_palette) +
        labs(title='3mer Fractions by Repeat Elements',
             x='3-mer',
             y='Fraction')
        )

ggsavefig_and_show(plot, "kmer3_canonical_sbs96_sbs_count_diff_repeats")

combined_df_fraction.to_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/RepeatMasker/kmer3_canonical_sbs_fraction_by_repeat.tsv", sep="\t")

adjusted_df = pd.DataFrame(index=df_canonical_sbs96_fraction.index)

for column in combined_df_fraction.columns:
    adjusted_df[column] = df_canonical_sbs96_fraction['count'] / combined_df_fraction[column]

# %%
# INFO: K-mer normalization factors
outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/RepeatMasker"
for rc in repeat_category:
    if rc == "DNA?":
        adjusted_df[rc].to_csv(f"{outdir}/kmer3_norm_factor_DNAqmark.tsv", sep="\t", header=None)
    else:
        adjusted_df[rc].to_csv(f"{outdir}/kmer3_norm_factor_{rc}.tsv", sep="\t", header=None)

# NOTE: K-mer normalization factors for total mutational (just 1s)
pd.DataFrame([1]*32, index=adjusted_df.index).to_csv(f"{outdir}/kmer3_norm_factor_all.tsv", sep="\t", header=None)



##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################


# INFO: Now Run VCF2SPECTRUM with and without k-mer normalization

# %%
desired_order = ['COLO829TB_Shared_Filtered_Pruned', 'COLO829TB_Shared_Filtered_Pruned_None_RE']

# INFO: Before k-mer adjustment
raw_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned*_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

remove_for_now = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned_Satellite_*_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))
remove_for_now.extend(sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned_*CDR*_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt")))

remove_for_now.extend(sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_*_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt")))

raw_files = set(raw_files) - set(remove_for_now)
raw_files= sorted(list(raw_files))

raw_files_dataframes = []
for raw_file in raw_files:
   df = pd.read_csv(raw_file, sep='\t', index_col=0).T
   raw_files_dataframes.append(df)

raw_assignment_table = pd.concat(raw_files_dataframes, axis=1)

remaining_columns = [col for col in raw_assignment_table.columns if col not in desired_order]
final_order = desired_order + remaining_columns
raw_assignment_table = raw_assignment_table[final_order]

raw_assignment_table_fraction = raw_assignment_table.div(raw_assignment_table.sum(axis=0), axis=1)
raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction[(raw_assignment_table_fraction != 0).any(axis=1)]

# NOTE: Rename Column names
column_rename_dict = dict(zip(raw_assignment_table_fraction_nonzero.columns, list(map(lambda x: x.replace("COLO829TB_Shared_Filtered_Pruned", "").lstrip('_'), raw_assignment_table_fraction_nonzero.columns))))
column_rename_dict["COLO829TB_Shared_Filtered_Pruned"] = "All"

raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction_nonzero.rename(columns=column_rename_dict)

# INFO: After k-mer adjustment
kmer_adjusted_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

remove_for_now = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned_Satellite_*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))
remove_for_now.extend(sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned_*CDR*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt")))
remove_for_now.extend(sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt")))

kmer_adjusted_files = set(kmer_adjusted_files) - set(remove_for_now)
kmer_adjusted_files= sorted(list(kmer_adjusted_files))

kmer_adjusted_files_dataframes = []
for kmer_adjusted_file in kmer_adjusted_files:
   df = pd.read_csv(kmer_adjusted_file, sep='\t', index_col=0).T
   kmer_adjusted_files_dataframes.append(df)

kmer_adjusted_assignment_table = pd.concat(kmer_adjusted_files_dataframes, axis=1)

remaining_columns = [col for col in kmer_adjusted_assignment_table.columns if col not in desired_order]
final_order = desired_order + remaining_columns
kmer_adjusted_assignment_table = kmer_adjusted_assignment_table[final_order]

kmer_adjusted_assignment_table_fraction = kmer_adjusted_assignment_table.div(kmer_adjusted_assignment_table.sum(axis=0), axis=1)
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction[(kmer_adjusted_assignment_table_fraction != 0).any(axis=1)]

# NOTE: Rename Column names
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction_nonzero.rename(columns=column_rename_dict)

# INFO: Make a proportional plots
distinct_sbs = set(raw_assignment_table_fraction_nonzero.index).union(set(kmer_adjusted_assignment_table_fraction_nonzero.index))
# NOTE: {'SBS1', 'SBS10b', 'SBS38', 'SBS5', 'SBS53', 'SBS7a', 'SBS7b', 'SBS7d', 'SBS87', 'SBS97'}

distinct_sbs_list = sorted(list(distinct_sbs))[::-1]
hex_colors = sns.color_palette("tab20", len(distinct_sbs_list)).as_hex()
#sbs_color_dict = dict(zip(distinct_sbs_list, hex_colors))

df_melted = raw_assignment_table_fraction_nonzero.reset_index()
df_long = df_melted.melt(id_vars=['index'], 
                         var_name='Sample_Type', 
                         value_name='Proportion')
df_long = df_long.rename(columns={'index': 'SBS_Signature'})
df_long['Sample_Type_Clean'] = df_long['Sample_Type'].str.replace('COLO829TB_Shared_Filtered_', '', regex=False)
df_long['Sample_Type_Clean'] = df_long['Sample_Type_Clean'].str.replace('COLO829TB_Shared_Filtered', 'All', regex=False)

desired_order = ['All', 'None_RE', 'Satellite', 
                 'Low_complexity', 'Simple_repeat', 
                 'Retroposon', 'LTR', 'LINE',  
                 'SINE', 'DNA']

df_long['Sample_Type_Clean'] = pd.Categorical(df_long['Sample_Type_Clean'], 
                                              categories=desired_order, 
                                              ordered=True)

desired_sbs_order = ['SBS1', 'SBS5', 'SBS7a', 'SBS7b', 'SBS7d', 'SBS38', 'SBS97']
df_long['SBS_Signature'] = pd.Categorical(df_long['SBS_Signature'], categories=desired_sbs_order, ordered=True)

plot1 = (ggplot(df_long, aes(x='Sample_Type_Clean', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     scale_x_discrete(limits=reversed) + 
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Pre-Normalization)',
          x='Genomic Context',
          y='Percentage') +
     theme_minimal() +
     theme(
           text=element_text(family='Arial'),
           legend_position='right',
           figure_size=(12, 8),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=14, color='black'),
           axis_title_y=element_text(size=14, color='black'),
           axis_text_x=element_text(rotation=0, size=10, color='black'),
           axis_text_y=element_text(rotation=0, size=10, color='black')
           ) +
     coord_flip()
)

ggsavefig_and_show(plot1, "Raw_SBS_Assignment")

df_melted = kmer_adjusted_assignment_table_fraction_nonzero.reset_index()
df_long = df_melted.melt(id_vars=['index'], 
                         var_name='Sample_Type', 
                         value_name='Proportion')
df_long = df_long.rename(columns={'index': 'SBS_Signature'})
df_long['Sample_Type_Clean'] = df_long['Sample_Type'].str.replace('COLO829TB_Shared_Filtered_', '', regex=False)
df_long['Sample_Type_Clean'] = df_long['Sample_Type_Clean'].str.replace('COLO829TB_Shared_Filtered', 'All', regex=False)

df_long['Sample_Type_Clean'] = pd.Categorical(df_long['Sample_Type_Clean'], 
                                              categories=desired_order, 
                                              ordered=True)

df_long['SBS_Signature'] = pd.Categorical(df_long['SBS_Signature'], categories=desired_sbs_order, ordered=True)

plot2 = (ggplot(df_long, aes(x='Sample_Type_Clean', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     scale_x_discrete(limits=reversed) + 
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Percentage') +
     theme_minimal() +
     theme(
           text=element_text(family='Arial'),
           legend_position='right',
           figure_size=(12, 8),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=14, color='black'),
           axis_title_y=element_text(size=14, color='black'),
           axis_text_x=element_text(rotation=0, size=10, color='black'),
           axis_text_y=element_text(rotation=0, size=10, color='black')
           ) +
     coord_flip()
)

ggsavefig_and_show(plot2, "Kmer-adjusted_SBS_Assignment")

# %%
# INFO: I don't like the output of SigProfiler Plotting so I'm making my own (will make it in general fashion)
# INFO: SBS96 for Total SNVs:

sbs6_colors = {'C>A': '#03bcee',
          'C>G': 'black',
          'C>T': '#e32926',
          'T>A': '#cac9c9',
          'T>C': '#a1ce63',
          'T>G': '#ebc6c4'}

sbs6_order = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

sbs_matrix=pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_SBS96/Kmer_normalization/COLO829TB_Shared_Filtered_Pruned.SBS96.kmer_normalized.all", sep="\t")
sbs_matrix['sbs6'] = sbs_matrix['MutationType'].str.extract(r'\[([ACGT]>[ACGT])\]')

sbs_matrix['sbs6'] = pd.Categorical(
                        sbs_matrix['sbs6'], 
                        categories=sbs6_order, 
                        ordered=True)

sbs_matrix = sbs_matrix.sort_values(['sbs6', 'MutationType']).reset_index(drop=True)

sbs_matrix['MutationType'] = pd.Categorical(
    sbs_matrix['MutationType'], 
    categories=sbs_matrix['MutationType'].tolist(),
    ordered=True)

sbs_matrix["Percentage"] = sbs_matrix['COLO829TB_Shared_Filtered_Pruned']*100 / sbs_matrix['COLO829TB_Shared_Filtered_Pruned'].sum()

plot = (ggplot(sbs_matrix, aes(x='MutationType', y='COLO829TB_Shared_Filtered_Pruned', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Count') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_kmer_normalized")

plot = (ggplot(sbs_matrix, aes(x='MutationType', y='Percentage', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Percentage') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_kmer_normalized_percentage")

# %%
# INFO: DBS78 for Total SNVs:
dbs10_colors = {"AC": "#03bcee",
                "AT": "#0366cb",
                "CC": "#a1ce63",
                "CG": "#016601",
                "CT": "#fe9898",
                "GC": "#e32926",
                "TA": "#feb166",
                "TC": "#fe8001",
                "TG": "#cb98fe",
                "TT": "#4c0198"} # 10 "Source" Doublets
dbs10_order = ['AC', 'AT', 'CC', 'CG', 'CT', 'GC', 'TA', 'TC', 'TG', 'TT']

dbs_matrix=pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_DBS78/COLO829TB_Shared_Filtered_Pruned.DBS78.all", sep="\t")
dbs_matrix['dbs10'] = dbs_matrix['MutationType'].str[:2]

dbs_matrix['dbs10'] = pd.Categorical(
                        dbs_matrix['dbs10'], 
                        categories=dbs10_order, 
                        ordered=True)

dbs_matrix = dbs_matrix.sort_values(['dbs10', 'MutationType']).reset_index(drop=True)

dbs_matrix['MutationType'] = pd.Categorical(
    dbs_matrix['MutationType'], 
    categories=dbs_matrix['MutationType'].tolist(),
    ordered=True)

dbs_matrix["Percentage"] = dbs_matrix['COLO829TB_Shared_Filtered_Pruned']*100 / dbs_matrix['COLO829TB_Shared_Filtered_Pruned'].sum()

plot = (ggplot(dbs_matrix, aes(x='MutationType', y='COLO829TB_Shared_Filtered_Pruned', fill='dbs10')) +
        geom_col() +
        scale_fill_manual(values=dbs10_colors) +
        labs(title='DBS78 Mutation Spectrum',
             x='Mutation Type',
             y='Count') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "dbs78_spectrum_colotb_snvs_shared_final_filtered_pruned")

plot = (ggplot(dbs_matrix, aes(x='MutationType', y='Percentage', fill='dbs10')) +
        geom_col() +
        scale_fill_manual(values=dbs10_colors) +
        labs(title='DBS78 Mutation Spectrum',
             x='Mutation Type',
             y='Percentage') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "dbs78_spectrum_colotb_snvs_shared_final_filtered_pruned_percentage")

# %% 
# INFO: Make Mutational Signature Decomposition Plot only for Total SNVs
kmer_adjusted_assignment_table_fraction_nonzero_onlytotal = kmer_adjusted_assignment_table_fraction_nonzero[["All"]].copy()
kmer_adjusted_assignment_table_fraction_nonzero_onlytotal = kmer_adjusted_assignment_table_fraction_nonzero_onlytotal[(kmer_adjusted_assignment_table_fraction_nonzero_onlytotal != 0).any(axis=1)]

df_melted = kmer_adjusted_assignment_table_fraction_nonzero_onlytotal.reset_index()
df_long = df_melted.melt(id_vars=['index'], 
                         var_name='Sample_Type', 
                         value_name='Proportion')
df_long = df_long.rename(columns={'index': 'SBS_Signature'})

sbs_order = df_long['SBS_Signature'].tolist()  # This will be ['SBS1', 'SBS7a', 'SBS7b', 'SBS38']
df_long['SBS_Signature'] = pd.Categorical(
    df_long['SBS_Signature'], 
    categories=sbs_order, 
    ordered=True
)

plot = (ggplot(df_long, aes(x='Sample_Type', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.2) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     scale_x_discrete(limits=reversed) + 
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Percentage') +
     theme_tufte() +
     theme(text=element_text(family='Arial'),
           axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
           axis_text_y=element_text(color='black'),
           axis_line_y=element_line(size=0.5, color='black'),
           axis_ticks_major=element_line(size=0.5, color='black'),
           plot_title=element_text(size=10, color='black'),
           axis_title=element_text(size=10, color='black'),
           legend_position='right',
           figure_size=(4, 2))
    )           
ggsavefig_and_show(plot, "Kmer-adjusted_SBS_Assignment_for_Total_only")



# %%
# INFO: SBS96 Mutational Counts Plot for Repeat Elements
sbs6_colors = {'C>A': '#03bcee',
          'C>G': 'black',
          'C>T': '#e32926',
          'T>A': '#cac9c9',
          'T>C': '#a1ce63',
          'T>G': '#ebc6c4'}

sbs6_order = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']


rc_category = ["COLO829TB_Shared_Filtered_Pruned_None_RE",
               "COLO829TB_Shared_Filtered_Pruned_Simple_repeat",
               "COLO829TB_Shared_Filtered_Pruned_DNA",
               "COLO829TB_Shared_Filtered_Pruned_SINE",
               "COLO829TB_Shared_Filtered_Pruned_Retroposon",
               "COLO829TB_Shared_Filtered_Pruned_Low_complexity",
               "COLO829TB_Shared_Filtered_Pruned_Satellite",
               "COLO829TB_Shared_Filtered_Pruned_LINE",
               "COLO829TB_Shared_Filtered_Pruned_LTR"]

for rc in rc_category:

    rc_name = '_'.join(rc.split('_')[4:])

    sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/{rc}_SBS96/Kmer_normalization/{rc}.SBS96.kmer_normalized.all", sep="\t")
    sbs_matrix['sbs6'] = sbs_matrix['MutationType'].str.extract(r'\[([ACGT]>[ACGT])\]')
    
    sbs_matrix['sbs6'] = pd.Categorical(
                                sbs_matrix['sbs6'], 
                                categories=sbs6_order, 
                                ordered=True)

    sbs_matrix = sbs_matrix.sort_values(['sbs6', 'MutationType']).reset_index(drop=True)

    sbs_matrix['MutationType'] = pd.Categorical(
                                    sbs_matrix['MutationType'], 
                                    categories=sbs_matrix['MutationType'].tolist(),
                                    ordered=True)

    sbs_matrix["Percentage"] = sbs_matrix[f'{rc}']*100 / sbs_matrix[f'{rc}'].sum()

    plot = (ggplot(sbs_matrix, aes(x='MutationType', y='Percentage', fill='sbs6')) +
            geom_col() +
            scale_fill_manual(values=sbs6_colors) +
            labs(title=f'{rc_name}',
                 x='Mutation Type',
                 y='Percentage') +
            theme_tufte() +
            theme(figure_size=(8, 2.5),
                  text=element_text(family='Arial'),
                  axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
                  axis_text_y=element_text(color='black'),
                  plot_title=element_text(size=12, color='black', hjust=0.5),
                  axis_line_x=element_line(size=0.5, color='black'),
                  axis_line_y=element_line(size=0.5, color='black'),
                  axis_ticks_major=element_line(size=0.2, color='black'),
                  axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
                  axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
            )
    
    ggsavefig_and_show(plot, f"sbs96_spectrum_{rc}_kmer_normalized_percentage")











# %%
cos_similarity_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned*SBS96/Assignment_Solution/Solution_Stats/Assignment_Solution_Samples_Stats.txt"))
























# %%
snv_amsd_result = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/merged_amsd_results_09132025.tsv", sep="\t")

snv_amsd_result = snv_amsd_result[(snv_amsd_result['comparison'] == "vs non-repeat") & (snv_amsd_result["group"] != "None_RE")].reset_index(drop=True)
snv_amsd_result = apply_bh_correction(snv_amsd_result, ["pvalue"])

snv_amsd_result['-log10pvalue_BH'] = -np.log10(snv_amsd_result['pvalue_BH'])

adjust_text_dict = {
    'expand': (1, 4.5),
    'arrowprops': {
        'arrowstyle': '-'
        }
    }

plot = (
        ggplot(snv_amsd_result, aes(x='cosine_dist', y='-log10pvalue_BH')) +
        geom_point(size=1, alpha=0.7, color='black') +
        geom_hline(yintercept=-np.log10(0.05), color='red', linetype='dotted', size=0.4) +
        geom_hline(yintercept=-np.log10(10**(-5)), color='red', linetype='dashed', size=0.3) +
        theme_minimal() +
        theme(figure_size=(4, 3),
              text=element_text(family='Arial'),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_title_x=element_text(size=6),
              axis_title_y=element_text(size=6),
              axis_text_x=element_text(rotation=0, hjust=0.5, size=6, color='black'),
              axis_text_y=element_text(rotation=0, hjust=0.5, size=6, color='black'),
              axis_ticks_major_x=element_line(size=1.5, color='black'),
              axis_ticks_major_y=element_line(size=0.5, color='black'),
              panel_grid_major_x=element_line(size=0.5),
              panel_grid_major_y=element_line(size=0.5),
              panel_grid_minor_x=element_line(size=0.25),
              panel_grid_minor_y=element_line(size=0.25),
              axis_ticks_minor_x=element_line(size=0.01, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'),
            ) +
        geom_text(aes(label='group'), 
                  adjust_text=adjust_text_dict,
                  size=8) +
        labs(title='',
             x='Cosine distance between SBS Mutational Spectrum of Non-repeat elements',
             y='-log10(Benjamini-Hochberg P value)'
             ) +
        xlim(-0.00001, None) +
        ylim(0, None)
        )

ggsavefig_and_show(plot, "snv_amsd_results_cosine_dist_vs_neglog10pvalue_BH")

# %%
# INFO: Mutational Rate across different repeat elements / segmentally duplicated regions and etc.
rc_length = dict()
for rc in set(repeatmasker_pr.Name1.values): # NOTE: rf => RepeatMasker Class
    rc_length[rc] = repeatmasker_pr[repeatmasker_pr.Name1 == rc].merge().length

rc_length["None_RE"] = callable_pr.subtract(repeatmasker_pr).length

rc_length_df = pd.DataFrame.from_dict(rc_length, orient='index', columns=["Length"])

rc_snv_count = dict()
for rc in repeat_category[1:]:
    if rc == "DNA?":
       pattern = r"DNA\?"
    else:
        pattern = r"{}".format(rc)

    count = colotb_snvs_shared_final_filtered_pruned[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").str.contains(pattern, na=False)].shape[0]
    rc_snv_count[rc] = count

rc_snv_count["None_RE"] = colotb_snvs_shared_final_filtered_pruned[vcf_info_getter(colotb_snvs_shared_final_filtered_pruned, "RM").isna()].shape[0]

rc_snv_count_df = pd.DataFrame.from_dict(rc_snv_count, orient='index', columns=["SNV_Count"])

rc_length_snv_count_df = pd.concat([rc_length_df, rc_snv_count_df], axis=1)
rc_length_snv_count_df["SNV_rate"] = (rc_length_snv_count_df["SNV_Count"] / rc_length_snv_count_df["Length"])

total_length = callable_pr.length
total_snv_count = colotb_snvs_shared_final_filtered_pruned.shape[0]
total_snv_rate = total_snv_count / total_length

total_snv = {"Length": total_length, 
             "SNV_Count": total_snv_count, 
             "SNV_rate": total_snv_rate}

rc_length_snv_count_df = pd.concat([rc_length_snv_count_df, pd.DataFrame.from_dict(total_snv, orient='index', columns=['Total']).T])
rc_length_snv_count_df = rc_length_snv_count_df.sort_values(by="Length", ascending=False)
rc_length_snv_count_df["SNV_rate"] = pd.to_numeric(rc_length_snv_count_df["SNV_rate"])

#del total_snv, total_snv_count, total_snv_rate, total_length, rc_length_df, rc_length

#Length	SNV_Count	SNV_rate
#Total	4735201678	44810	9.5e-06
#None_RE	2258123835	17663	7.8e-06
#LINE	980369744	9725	9.9e-06
#SINE	619662063	4756	7.7e-06
#LTR	426000532	4525	1.1e-05
#Satellite	185390929	5498	3.0e-05
#DNA	171566262	1244	7.3e-06
#Simple_repeat	71885443	1066	1.5e-05
#Retroposon	13667695	187	1.4e-05
#Low_complexity	9750594	149	1.5e-05
#Unknown	1470650	12	8.2e-06
#snRNA	722619	4	5.5e-06
#RC	713849	10	1.4e-05
#srpRNA	479183	2	4.2e-06
#rRNA	345353	9	2.6e-05
#scRNA	291894	1	3.4e-06
#tRNA	172972	12	6.9e-05
#DNA?	46321	0	0.0e+00

rc_length_snv_count_df_filtered = rc_length_snv_count_df[rc_length_snv_count_df["SNV_Count"] > 100]
rc_length_snv_count_df_filtered["SNV_enrichment"] = (rc_length_snv_count_df_filtered["SNV_rate"]) / total_snv_rate
rc_length_snv_count_df_filtered["SNV_enrichment_over_None_RE"] = (rc_length_snv_count_df_filtered["SNV_rate"]) / (rc_length_snv_count_df_filtered["SNV_rate"].loc["None_RE"])

# %%
total_row = rc_length_snv_count_df_filtered[rc_length_snv_count_df_filtered.index == 'Total']
none_re_row = rc_length_snv_count_df_filtered[rc_length_snv_count_df_filtered.index == 'None_RE']
other_rows = rc_length_snv_count_df_filtered[~rc_length_snv_count_df_filtered.index.isin(['Total', 'None_RE'])]

other_rows_sorted = other_rows.sort_values('SNV_enrichment', ascending=False)

rc_length_snv_count_df_filtered = pd.concat([total_row, none_re_row, other_rows_sorted])

del total_row, none_re_row, other_rows, other_rows_sorted
# %%
def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_tuple[0] * 255),
        int(rgb_tuple[1] * 255),
        int(rgb_tuple[2] * 255)
    )

rc_color_palette = sns.color_palette(palette='terrain', n_colors=rc_length_snv_count_df_filtered.shape[0])
rc_color_map = {key: color for key, color in zip(rc_length_snv_count_df_filtered.index.values, rc_color_palette)}
rc_color_map_hex = {category: rgb_to_hex(color) for category, color in rc_color_map.items()}


rc_length_snv_count_df_filtered_plot = rc_length_snv_count_df_filtered.reset_index()
rc_length_snv_count_df_filtered_plot.columns = ['Category'] + list(rc_length_snv_count_df_filtered_plot.columns[1:])

plot = (ggplot(rc_length_snv_count_df_filtered_plot, aes(x='Category', y='SNV_rate', color='Category')) +
        geom_point(size=3) +
        coord_flip() +
        scale_x_discrete(limits=rc_length_snv_count_df_filtered_plot['Category'][::-1]) +
        scale_color_manual(values=rc_color_map_hex) +
        labs(title='SNV Rate across different Repeat Elements',
             x='',
             y='SNV Rate') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text=element_text(color='black')))

ggsavefig_and_show(plot, "snv_rate_across_re")

plot = (ggplot(rc_length_snv_count_df_filtered_plot, aes(x='Category', y='SNV_enrichment', fill='Category')) +
        geom_col() +
        coord_flip() +
        scale_x_discrete(limits=rc_length_snv_count_df_filtered_plot['Category'][::-1]) +
        scale_fill_manual(values=rc_color_map_hex) +
        labs(title='SNV Enrichment across different Repeat Elements',
             x='',
             y='SNV Enrichment (Obs/Exp ratio)') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text=element_text(color='black')))

ggsavefig_and_show(plot, "snv_enrichment_across_re")

plot = (ggplot(rc_length_snv_count_df_filtered_plot[1:], aes(x='Category', y='SNV_enrichment_over_None_RE', fill='Category')) +
        geom_col() +
        coord_flip() +
        scale_x_discrete(limits=rc_length_snv_count_df_filtered_plot[1:]['Category'][::-1]) +
        scale_fill_manual(values=rc_color_map_hex) +
        labs(title='SNV Enrichment across different Repeat Elements',
             x='',
             y='SNV Enrichment (over SNV rate in non-repetitive regions)') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text=element_text(color='black')))

ggsavefig_and_show(plot, "snv_enrichment_across_re_over_none-re")

# %%
rc_length_snv_count_df_filtered_plot['Category'] = pd.Categorical(
    rc_length_snv_count_df_filtered_plot['Category'], 
    categories=rc_length_snv_count_df_filtered_plot['Category'].tolist()[::-1], 
    ordered=True
)

plot = (
    ggplot(rc_length_snv_count_df_filtered_plot, aes(x='Category', y='SNV_rate', size='SNV_Count')) +
    geom_point() +
    labs(
        title='',
        x='Genomic Context',
        y='Mutation Rate',
        size='# of SNV'
    ) +
    theme_minimal() +
    theme(
        figure_size=(4.5,3),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', hjust=0.5, size=8),
        axis_text_y=element_text(color='black'),
        plot_title=element_text(size=12, color='black', hjust=0.5),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.2, color='black'),
        axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.02, color='darkgray')
        ) +
    coord_flip()
    )

ggsavefig_and_show(plot, "snv_rate_across_repeat_dotplot")

# %%
# %%
plot = (
    ggplot(rc_length_snv_count_df_filtered_plot[rc_length_snv_count_df_filtered_plot["Category"] != "Total"], aes(x='Category', y='SNV_rate', size='SNV_Count')) +
    geom_point() +
    labs(
        title='',
        x='Genomic Context',
        y='Mutation Rate',
        size='# of SNV'
    ) +
    theme_minimal() +
    theme(
        figure_size=(4.5, 3),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', hjust=0.5, size=8),
        axis_text_y=element_text(color='black'),
        plot_title=element_text(size=12, color='black', hjust=0.5),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.2, color='black'),
        axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.02, color='darkgray')
        ) +
    coord_flip()
    )

ggsavefig_and_show(plot, "snv_rate_across_repeat_wo_total_dotplot")
# %%
# INFO: Re-defining Satellite Elements
# NOTE: Manually curating Satellite sub-families
# NOTE:  Nick Altemose's comment "SAR is HSat1A, and HSATI is HSat1B (found predominantly on chrY). They are the most A/T rich sequences in the genome, at ~80% AT. So they might have an apparently lower mutation rate than other satellites simply due to a lack of cytosines. Gamma satellite, on the other hand, is one of the most GC rich satellites. HSat2 (“HSATII”, derived from “CATTCGATTC”) and HSat3 (CATTC/GAATG) are full of dipyrimidines on one strand."
satellite_unique = sorted(repeatmasker_pr[repeatmasker_pr.Name1 == "Satellite"].Name3.unique())
satellite_replace_list = ['HSat3',
                          'HSat3',
                          'ACRO1',
                          'α-Satellite',
                          'BSR_Beta',
                          'CER',
                          'D20S16',
                          'GSAT',
                          'GSATII',
                          'GSATX',
                          'HSat4',
                          'HSat5',
                          'HSat6',
                          'HSat1B',
                          'HSat2',
                          'LSAU',
                          'MSR1',
                          'HSat1A',
                          'SATR1',
                          'SATR2',
                          'SST1',
                          'TAR1']

sa_replace_dict = dict(zip(satellite_unique, satellite_replace_list))

sa_pr = repeatmasker_pr[repeatmasker_pr.Name1 == "Satellite"].copy()
sa_pr.Name4 = sa_pr.Name3.replace(sa_replace_dict)

satellite_unique_new = sorted(sa_pr.Name4.unique())

# INFO: Analysis On Satellites
sa_length_dict = dict()
sa_snv_count_dict = dict()
sa_snv_vcf_dict = dict()

for sa in satellite_unique_new:
    sa_pr_subset = sa_pr[sa_pr.Name4 == sa].merge().copy()
    sa_length_dict[sa] = sa_pr_subset.length
    
    snvid_in_interval = list()

    for i, interval in sa_pr_subset.df.iterrows():
        mask = (
            (colotb_snvs_shared_final_filtered_pruned['CHROM'] == interval['Chromosome']) &
            (colotb_snvs_shared_final_filtered_pruned['POS'] > interval['Start']) &
            (colotb_snvs_shared_final_filtered_pruned['POS'] <= interval['End'])
        )

        snvid_in_interval.extend(colotb_snvs_shared_final_filtered_pruned[mask]["SNVid"].values)

    sa_snv_count_dict[sa] = len(set(snvid_in_interval))
    sa_snv_vcf_dict[sa] = colotb_snvs_shared_final_filtered_pruned[colotb_snvs_shared_final_filtered_pruned["SNVid"].isin(snvid_in_interval)].copy()

sa_length_df = pd.DataFrame.from_dict(sa_length_dict, orient='index')
sa_length_df.columns = ["Length"]
sa_length_df = sa_length_df.sort_values(by="Length", ascending=False)

sa_snv_count_df = pd.DataFrame.from_dict(sa_snv_count_dict, orient='index')
sa_snv_count_df.columns = ["SNV_Count"]

sa_df = pd.concat([sa_length_df, sa_snv_count_df], axis=1)
sa_df["SNV_rate"] = sa_df["SNV_Count"] / sa_df["Length"]

sa_df_plot = sa_df.reset_index()
sa_df_plot.columns = ['Category'] + list(sa_df_plot.columns[1:])

# %%
sa_color_palette = sns.color_palette(palette='tab20', n_colors=sa_df.shape[0])
sa_color_map = {key: color for key, color in zip(sa_df.index.values, sa_color_palette)}
sa_color_map_hex = {category: rgb_to_hex(color) for category, color in sa_color_map.items()}

plot = (ggplot(sa_df_plot, aes(x='Category', y='SNV_rate', color='Category')) +
        geom_point(size=3) +
        coord_flip() +
        scale_x_discrete(limits=sa_df_plot['Category'][::-1]) +
        scale_color_manual(values=sa_color_map_hex) +
        labs(title='SNV Rate across different Satellite Family',
             x='',
             y='SNV Rate') +
        theme_minimal() +
        theme(axis_text=element_text(color='black')))

ggsavefig_and_show(plot, "snv_rate_across_satellites")

# %%
sa_color_palette = sns.color_palette(palette='tab20', n_colors=sa_df.shape[0])
sa_color_map = {key: color for key, color in zip(sa_df.index.values, sa_color_palette)}
sa_color_map_hex = {category: rgb_to_hex(color) for category, color in sa_color_map.items()}

plot = (ggplot(sa_df_plot, aes(x='Category', y='SNV_rate', fill='Category')) +
        geom_col() +
        coord_flip() +
        scale_x_discrete(limits=sa_df_plot['Category'][::-1]) +
        scale_color_manual(values=sa_color_map_hex) +
        labs(title='SNV Rate across different Satellite Family',
             x='',
             y='SNV Rate') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text_x=element_text(color='black', hjust=0.5),
              axis_text_y=element_text(color='black', hjust=1),
              legend_position='none')
        )

ggsavefig_and_show(plot, "snv_rate_across_satellites_bar")

# %%
# INFO: Dot plot with SNV count size
# INFO: Filter satellite sub-families with SNV count >= 50
sa_df_plot_snv50 = sa_df_plot[sa_df_plot["SNV_Count"] >= 50]

plot = (
    ggplot(sa_df_plot_snv50, aes(x='Category', y='SNV_rate', size='SNV_Count')) +
    geom_point() +
    labs(
        title='',
        x='Satellite Subfamilies (n(SNV)≥50)',
        y='Mutation Rate',
        size='# of SNV'
    ) +
    theme_minimal() +
    theme(
        figure_size=(4,3),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', hjust=0.5, size=8),
        axis_text_y=element_text(color='black'),
        plot_title=element_text(size=12, color='black', hjust=0.5),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.2, color='black'),
        axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.02, color='darkgray')
        ) +
    coord_flip()
    )

ggsavefig_and_show(plot, "snv_rate_across_satellites_snv50_dotplot")

# %%
rc_length_snv_count_df_filtered_plot_with_sa = pd.concat([rc_length_snv_count_df_filtered_plot, sa_df_plot_snv50])

desired_order = ['Total',
                 'None_RE',
                 'Satellite',
                 'α-Satellite',
                 'HSat1A',
                 'HSat2',
                 'HSat3',
                 'BSR_Beta',
                 'Low_complexity',
                 'Simple_repeat',
                 'Retroposon',
                 'LTR',
                 'LINE',
                 'SINE',
                 'DNA']

rc_length_snv_count_df_filtered_plot_with_sa['Category'] = pd.Categorical(
    rc_length_snv_count_df_filtered_plot_with_sa['Category'], 
    categories=desired_order[::-1], 
    ordered=True
)

plot = (
    ggplot(rc_length_snv_count_df_filtered_plot_with_sa, aes(x='Category', y='SNV_rate', size='SNV_Count')) +
    geom_point() +
    labs(
        title='',
        x='Genomic Contexts',
        y='Mutation Rate',
        size='# of SNV'
    ) +
    theme_minimal() +
    theme(
        figure_size=(4.5,3),
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', hjust=0.5, size=8),
        axis_text_y=element_text(color='black'),
        plot_title=element_text(size=12, color='black', hjust=0.5),
        axis_line_x=element_line(size=0.5, color='black'),
        axis_line_y=element_line(size=0.5, color='black'),
        axis_ticks_major=element_line(size=0.2, color='black'),
        axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
        axis_ticks_minor_y=element_line(size=0.02, color='darkgray')
        ) +
    coord_flip()
    )

ggsavefig_and_show(plot, "snv_rate_across_repeat_and_satellites_snv50_dotplot")



# %%
# INFO: SBS96 Mutational Counts Plot for Satellite Sub-families (SNV≥50)
sbs6_colors = {'C>A': '#03bcee',
          'C>G': 'black',
          'C>T': '#e32926',
          'T>A': '#cac9c9',
          'T>C': '#a1ce63',
          'T>G': '#ebc6c4'}

sbs6_order = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

# NOTE: sa_df_plot_snv50["Category"].tolist() -> ['α-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3']
sa_category = ["COLO829TB_Shared_Filtered_Pruned_Satellite_alpha-Satellite", 
               "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat2", 
               "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat1A", 
               "COLO829TB_Shared_Filtered_Pruned_Satellite_BSR_Beta", 
               "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat3"]

for sa in sa_category:

    sa_name = '_'.join(sa.split('_')[4:])

    sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/{sa}_SBS96/Kmer_normalization/{sa}.SBS96.kmer_normalized.all", sep="\t")
    sbs_matrix['sbs6'] = sbs_matrix['MutationType'].str.extract(r'\[([ACGT]>[ACGT])\]')
    
    sbs_matrix['sbs6'] = pd.Categorical(
                                sbs_matrix['sbs6'], 
                                categories=sbs6_order, 
                                ordered=True)

    sbs_matrix = sbs_matrix.sort_values(['sbs6', 'MutationType']).reset_index(drop=True)

    sbs_matrix['MutationType'] = pd.Categorical(
                                    sbs_matrix['MutationType'], 
                                    categories=sbs_matrix['MutationType'].tolist(),
                                    ordered=True)

    sbs_matrix["Percentage"] = sbs_matrix[f'{sa}']*100 / sbs_matrix[f'{sa}'].sum()

    plot = (ggplot(sbs_matrix, aes(x='MutationType', y='Percentage', fill='sbs6')) +
            geom_col() +
            scale_fill_manual(values=sbs6_colors) +
            labs(title=f'{sa_name}',
                 x='Mutation Type',
                 y='Percentage') +
            theme_tufte() +
            theme(figure_size=(8, 2.5),
                  text=element_text(family='Arial'),
                  axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
                  axis_text_y=element_text(color='black'),
                  plot_title=element_text(size=12, color='black', hjust=0.5),
                  axis_line_x=element_line(size=0.5, color='black'),
                  axis_line_y=element_line(size=0.5, color='black'),
                  axis_ticks_major=element_line(size=0.2, color='black'),
                  axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
                  axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
            )

    ggsavefig_and_show(plot, f"sbs96_spectrum_{sa}_kmer_normalized_percentage")

# %%
# INFO: Relationship between Relative difference in mutational rate between REs and Non-RE and Cosine distance between REs and Non-RE
rc_length_snv_count_df_filtered_plot_with_sa_wo_total = rc_length_snv_count_df_filtered_plot_with_sa[rc_length_snv_count_df_filtered_plot_with_sa["Category"] != "Total"].iloc[:, :4].reset_index(drop=True)

rc_length_snv_count_df_filtered_plot_with_sa_wo_total["Relative_enrichment_against_None_RE"] = rc_length_snv_count_df_filtered_plot_with_sa_wo_total["SNV_rate"] / rc_length_snv_count_df_filtered_plot_with_sa_wo_total[rc_length_snv_count_df_filtered_plot_with_sa_wo_total["Category"] == "None_RE"]["SNV_rate"].values

# INFO: Calculating cosine distance between None-RE and each RE (included in the above table)
# NOTE: None_RE vectors:
none_re_sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_None_RE_SBS96/Kmer_normalization/COLO829TB_Shared_Filtered_Pruned_None_RE.SBS96.kmer_normalized.all", sep="\t")

none_re_sbs_matrix["Percentage"] = none_re_sbs_matrix["COLO829TB_Shared_Filtered_Pruned_None_RE"]*100 / none_re_sbs_matrix["COLO829TB_Shared_Filtered_Pruned_None_RE"].sum()

# NOTE: RE Vectors that needs to be compared with None_RE vectors:
re_of_interest = ['Satellite',
                  'Low_complexity',
                  'Simple_repeat',
                  'Retroposon',
                  'LTR',
                  'LINE',
                  'SINE',
                  'DNA',
                  'α-Satellite',
                  'HSat2',
                  'HSat1A',
                  'BSR_Beta',
                  'HSat3']

re_of_interest_prefix = ["COLO829TB_Shared_Filtered_Pruned_Satellite",
                         "COLO829TB_Shared_Filtered_Pruned_Low_complexity",
                         "COLO829TB_Shared_Filtered_Pruned_Simple_repeat",
                         "COLO829TB_Shared_Filtered_Pruned_Retroposon",
                         "COLO829TB_Shared_Filtered_Pruned_LTR",
                         "COLO829TB_Shared_Filtered_Pruned_LINE",
                         "COLO829TB_Shared_Filtered_Pruned_SINE",
                         "COLO829TB_Shared_Filtered_Pruned_DNA",
                         "COLO829TB_Shared_Filtered_Pruned_Satellite_alpha-Satellite",
                         "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat2",
                         "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat1A",
                         "COLO829TB_Shared_Filtered_Pruned_Satellite_BSR_Beta",
                         "COLO829TB_Shared_Filtered_Pruned_Satellite_HSat3"]

re_of_interest_dict = dict(zip(re_of_interest, re_of_interest_prefix))

re_of_interest_cosine_distance = dict()

for j, k in re_of_interest_dict.items():

    sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/{k}_SBS96/Kmer_normalization/{k}.SBS96.kmer_normalized.all", sep="\t")
    sbs_matrix["Percentage"] = sbs_matrix[f'{k}']*100 / sbs_matrix[f'{k}'].sum()

    re_of_interest_cosine_distance[j] = distance.cosine(none_re_sbs_matrix['Percentage'], sbs_matrix['Percentage'])

re_of_interest_cosine_distance_df = pd.DataFrame.from_dict(re_of_interest_cosine_distance, orient='index', columns=['Cosine_distance'])

# NOTE: Merging the cosine distance values to the main dataframe
rc_length_snv_count_df_filtered_plot_with_sa_wo_total = rc_length_snv_count_df_filtered_plot_with_sa_wo_total.merge(
                                                                                        re_of_interest_cosine_distance_df, 
                                                                                        left_on='Category', 
                                                                                        right_index=True, 
                                                                                        how='left'
                                                                                    )

#rc_length_snv_count_df_filtered_plot_with_sa_wo_total["Relative_enrichment_against_None_RE_log2"] = np.log2(rc_length_snv_count_df_filtered_plot_with_sa_wo_total["Relative_enrichment_against_None_RE"])

# %%
# INFO: Scatterplot using plotnine (Relative_enrichment_against_None_RE vs Cosine_distance)
rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot = rc_length_snv_count_df_filtered_plot_with_sa_wo_total.dropna(subset=['Cosine_distance'])
rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot['color'] = rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot['Category'].apply(
    lambda x: 'grey' if x in ['DNA', 'Low_complexity'] else 'maroon'
)
plot = (ggplot(rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot, aes(x='Relative_enrichment_against_None_RE', y='Cosine_distance')) +
        geom_point(aes(color='color'), size=3, alpha=0.7) +
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        scale_x_continuous(trans='log10') +
        labs(title='',
             x='Relative Enrichment Against None-repetitive regions',
             y='Cosine distance between None-repetitive regions') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "cosine_distance_vs_relative_enrichment_against_none-re")


# %%
# INFO: Pearson's Chi-squared test
# INFO: against total SNV rates 
chi_pvalue_against_total_snv_rate = dict()
chi_stat_against_total_snv_rate = dict()
for i, row in sa_df.iterrows():
    contingency_table = [
                     [row['SNV_Count'], total_snv_count - row['SNV_Count']],
                     [row['Length'], total_length - row['Length']]
                     ]
    res = chi2_contingency(contingency_table)
    chi_pvalue_against_total_snv_rate[i] = res.pvalue
    chi_stat_against_total_snv_rate[i] = res.statistic

chi_pvalue_against_total_snv_rate_df = pd.DataFrame.from_dict(chi_pvalue_against_total_snv_rate, orient='index', columns=['pvalue_total'])
chi_stat_against_total_snv_rate_df = pd.DataFrame.from_dict(chi_stat_against_total_snv_rate, orient='index', columns=['statistic_total'])

# INFO: against None_RE SNV rates
chi_pvalue_against_none_re_snv_rate = dict()
chi_stat_against_none_re_snv_rate = dict()
for i, row in sa_df.iterrows():
    contingency_table = [
                     [row['SNV_Count'], rc_length_snv_count_df_filtered["SNV_Count"].loc["None_RE"] - row['SNV_Count']],
                     [row['Length'], rc_length_snv_count_df_filtered["Length"].loc["None_RE"] - row['Length']]
                     ]
    res = chi2_contingency(contingency_table)
    chi_pvalue_against_none_re_snv_rate[i] = res.pvalue
    chi_stat_against_none_re_snv_rate[i] = res.statistic

chi_pvalue_against_none_re_snv_rate_df = pd.DataFrame.from_dict(chi_pvalue_against_none_re_snv_rate, orient='index', columns=['pvalue_none_re'])
chi_stat_against_none_re_snv_rate_df = pd.DataFrame.from_dict(chi_stat_against_none_re_snv_rate, orient='index', columns=['statistic_none_re'])

# INFO: against satellite SNV rates
chi_pvalue_against_satellite_snv_rate = dict()
chi_stat_against_satellite_snv_rate = dict()
for i, row in sa_df.iterrows():
    contingency_table = [
                     [row['SNV_Count'], rc_length_snv_count_df_filtered["SNV_Count"].loc["Satellite"] - row['SNV_Count']],
                     [row['Length'], rc_length_snv_count_df_filtered["Length"].loc["Satellite"] - row['Length']]
                     ]
    res = chi2_contingency(contingency_table)
    chi_pvalue_against_satellite_snv_rate[i] = res.pvalue
    chi_stat_against_satellite_snv_rate[i] = res.statistic

chi_pvalue_against_satellite_snv_rate_df = pd.DataFrame.from_dict(chi_pvalue_against_satellite_snv_rate, orient='index', columns=['pvalue_satellite'])
chi_stat_against_satellite_snv_rate_df = pd.DataFrame.from_dict(chi_stat_against_satellite_snv_rate, orient='index', columns=['statistic_satellite'])

sa_df_chi_stats = pd.concat([sa_df, 
                             chi_pvalue_against_total_snv_rate_df,
                             chi_stat_against_total_snv_rate_df, 
                             chi_pvalue_against_none_re_snv_rate_df, 
                             chi_stat_against_none_re_snv_rate_df,
                             chi_pvalue_against_satellite_snv_rate_df,
                             chi_stat_against_satellite_snv_rate_df], axis=1)
sa_df_chi_stats_bh = apply_bh_correction(sa_df_chi_stats, ["pvalue_none_re"], alpha=0.01)
sa_df_chi_stats_bh_plot = sa_df_chi_stats_bh.reset_index()
sa_df_chi_stats_bh_plot.columns = ['Category'] + list(sa_df_chi_stats_bh_plot.columns[1:])

# %%
plot = (ggplot(sa_df_chi_stats_bh_plot, aes(x='Category', y='SNV_rate', fill='significant_none_re_BH')) +
        geom_col() +
        coord_flip() +
        scale_x_discrete(limits=sa_df_chi_stats_bh_plot['Category'][::-1]) +
        scale_fill_manual(values={True: 'cyan', False: 'dimgray'}) +
        labs(title='SNV Rate across different Satellite Family',
             x='',
             y='SNV Rate') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text_x=element_text(color='black', hjust=0.5),
              axis_text_y=element_text(color='black', hjust=1),
              legend_position='none')
        )

ggsavefig_and_show(plot, "snv_rate_across_satellites_bar")

# %%
# INFO: Segmentally duplicated regions
colotb_snvs_shared_final_filtered_pruned_sd = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, dupmasker_pr)

contingency_table = [
                     [colotb_snvs_shared_final_filtered_pruned_sd.shape[0], colotb_snvs_shared_final_filtered_pruned.shape[0] - colotb_snvs_shared_final_filtered_pruned_sd.shape[0]],
                     [dupmasker_pr.length, (callable_pr.length - dupmasker_pr.length)]
                     ]

res = chi2_contingency(contingency_table)

total_snv_rate = rc_length_snv_count_df_filtered_plot[rc_length_snv_count_df_filtered_plot['Category'] == "Total"]["SNV_rate"].values.item()

print(f"Mutational rate of Segmentally Duplicated Regions: {colotb_snvs_shared_final_filtered_pruned_sd.shape[0] / dupmasker_pr.length} ({colotb_snvs_shared_final_filtered_pruned_sd.shape[0] / dupmasker_pr.length / total_snv_rate: 1f}-fold increase vs. genome-wide SNV rate)")
print("Chi-squared test for Segmentally Duplicated Regions vs Non-Segmentally Duplicated Regions")
print(f"Statistic: {res.statistic}, p-value: {res.pvalue}")

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################


# %%
# INFO: Alpha-HOR: CDR vs Non-CDR

cdr_bl_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/CG_Methylation/Analysis/CDR/bulk_cdr_bl_r9_medconf_live.bed", header=None, sep="\t") # NOTE: CDR in COLO829BL inferred from mCG of ONT R9 Data (overlapped with "Live" alpha-HOR)

cdr_bl_pr = pr.from_dict({
    'Chromosome': cdr_bl_bed.iloc[:, 0],
    'Start': cdr_bl_bed.iloc[:, 1],
    'End': cdr_bl_bed.iloc[:, 2],
})

del cdr_bl_bed

# INFO: CDR 100kb Padded
cdr_bl_100kb_extended_pr = cdr_bl_pr.extend(100_000).merge().intersect(callable_pr).intersect(repeatmasker_pr[repeatmasker_pr.Name3 == "ALR_Alpha"]).copy()
non_cdr_bl_100kb_extended_pr = repeatmasker_pr[repeatmasker_pr.Name3 == "ALR_Alpha"].merge().subtract(cdr_bl_100kb_extended_pr).copy()

# INFO: CDR 50kb Padded
cdr_bl_50kb_extended_pr = cdr_bl_pr.extend(50_000).merge().intersect(callable_pr).intersect(repeatmasker_pr[repeatmasker_pr.Name3 == "ALR_Alpha"]).copy()
non_cdr_bl_50kb_extended_pr = repeatmasker_pr[repeatmasker_pr.Name3 == "ALR_Alpha"].merge().subtract(cdr_bl_50kb_extended_pr).copy()

# %%
colotb_snvs_shared_final_filtered_pruned_cdr_bl_100kb = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, cdr_bl_100kb_extended_pr)
colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_100kb = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, non_cdr_bl_100kb_extended_pr)

colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, cdr_bl_50kb_extended_pr)
colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, non_cdr_bl_50kb_extended_pr)

# %%
make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_cdr_bl_100kb,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.CDR-BL_100kb",
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
)

make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_100kb,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.Non-CDR-BL_100kb",
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
)

make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.CDR-BL_50kb",
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
)

make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.Non-CDR-BL_50kb",
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
)

# %%
# INFO: Two-proportion Z-test for CDR SNV and Non-CDR SNV rates
cdr_noncdr_snv_vectors = np.array([colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb.shape[0], colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb.shape[0]])
cdr_noncdr_length_vectors = np.array([cdr_bl_50kb_extended_pr.length, non_cdr_bl_50kb_extended_pr.length])

z_stat, p_value = proportions_ztest(cdr_noncdr_snv_vectors, cdr_noncdr_length_vectors)

print(f"Two proportion Z-test P-value for SNV rates in CDR vs Non-CDR: {p_value:2e}")

# %%
# INFO: Generating K-mer Frequency Tables
cdr_bl_100kb_extended_pr_kmer_fraction = get_kmer_fraction_from_frequency_tab(get_kmer_frequency_from_interval(intervals=cdr_bl_100kb_extended_pr))
non_cdr_bl_100kb_extended_pr_kmer_fraction = get_kmer_fraction_from_frequency_tab(get_kmer_frequency_from_interval(intervals=non_cdr_bl_100kb_extended_pr))

cdr_bl_50kb_extended_pr_kmer_fraction = get_kmer_fraction_from_frequency_tab(get_kmer_frequency_from_interval(intervals=cdr_bl_50kb_extended_pr))
non_cdr_bl_50kb_extended_pr_kmer_fraction = get_kmer_fraction_from_frequency_tab(get_kmer_frequency_from_interval(intervals=non_cdr_bl_50kb_extended_pr))

cdr_non_cdr_50kb_extended_kmer_fraction_merged = pd.concat([cdr_bl_50kb_extended_pr_kmer_fraction, non_cdr_bl_50kb_extended_pr_kmer_fraction], axis=1)
cdr_non_cdr_50kb_extended_kmer_fraction_merged.columns = ['CDR', 'Non-CDR']

print(f"Cosine similarity of 3-mer fracion between CDR vs Non-CDR: {1 - distance.cosine(cdr_non_cdr_50kb_extended_kmer_fraction_merged['CDR'].values, cdr_non_cdr_50kb_extended_kmer_fraction_merged['Non-CDR'].values)}")

df_plot = cdr_non_cdr_50kb_extended_kmer_fraction_merged.reset_index()
df_long = df_plot.melt(id_vars='3mer', var_name='Region', value_name='Fraction')

heatmap = (
    ggplot(df_long, aes(x='3mer', y='Region', fill='Fraction')) +
    geom_tile(color='white', size=0.5) +
    scale_fill_cmap(cmap_name='cividis_r') +
    theme_minimal() +
    theme(
        text=element_text(family='Arial'),
        axis_text_x=element_text(color='black', rotation=90, hjust=0.5, vjust=0.5),
        axis_text_y=element_text(color='black'),
        plot_title=element_text(size=14, color='black', hjust=0.5),
        figure_size=(8, 2.5),
        aspect_ratio=1/16
    ) +
    labs(
        title='3-mer Frequencies: CDR vs Non-CDR',
        x='3-mer',
        y='',
        fill='Fraction'
    )
)

ggsavefig_and_show(heatmap, "3mer_frequencies_cdr_vs_non_cdr_50kb")

# %% 
# INFO: Generate 3-mer Normalization Factors
get_3mer_norm_factor(
    cdr_bl_100kb_extended_pr_kmer_fraction,
    df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_cdr_bl_100kb.tsv", sep="\t", header=None)

get_3mer_norm_factor(
    non_cdr_bl_100kb_extended_pr_kmer_fraction,
    df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_non_cdr_bl_100kb.tsv", sep="\t", header=None)

get_3mer_norm_factor(
    cdr_bl_50kb_extended_pr_kmer_fraction,
    df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_cdr_bl_50kb.tsv", sep="\t", header=None)

get_3mer_norm_factor(
    non_cdr_bl_50kb_extended_pr_kmer_fraction,
    df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_non_cdr_bl_50kb.tsv", sep="\t", header=None)

# %%
#cdr_bl_100kb_extended_snv_rate = colotb_snvs_shared_final_filtered_pruned_cdr_bl_100kb.shape[0] / cdr_bl_100kb_extended_pr.length
#non_cdr_bl_100kb_extended_snv_rate = colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_100kb.shape[0] / non_cdr_bl_100kb_extended_pr.length
# 2.8731612333195284e-05
# 2.3002183241334003e-05

#cdr_bl_50kb_extended_snv_rate = colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb.shape[0] / cdr_bl_50kb_extended_pr.length
#non_cdr_bl_50kb_extended_snv_rate = colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb.shape[0] / non_cdr_bl_50kb_extended_pr.length
# 3.1718070446118926e-05
# 2.3176262374774776e-05

# INFO: Sanity check: 
# colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb.shape[0] + colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb.shape[0] should be identical to `α-Satellite` `SNV_Count`` in `sa_df_plot`

cdr_bl_50kb_df = pd.DataFrame({"Length": [cdr_bl_50kb_extended_pr.length, non_cdr_bl_50kb_extended_pr.length], "SNV_Count": [colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb.shape[0], colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb.shape[0]]})

cdr_bl_50kb_extended_pr_kmer_fraction.columns = ["CDR"]
combined_df_fraction = pd.concat([combined_df_fraction, cdr_bl_50kb_extended_pr_kmer_fraction], axis=1)

non_cdr_bl_50kb_extended_pr_kmer_fraction.columns = ["Non-CDR"]
combined_df_fraction = pd.concat([combined_df_fraction, non_cdr_bl_50kb_extended_pr_kmer_fraction], axis=1)

# %%
# INFO: Make VCF for Satellite sub-families with SNV counts at least 10 (a-Satellite already taken care of above but this time as a whole)
# NOTE: First make pyranges intervals for each satellite sub-family
for sa in sa_df[sa_df["SNV_Count"] >= 10].index:
    sa_subfamily_pr = sa_pr[sa_pr.Name4 == sa].merge() # NOTE: Already callable regions
    sa_subfamily_vcf = vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, sa_subfamily_pr)

    if sa == 'α-Satellite':
        sa = 'alpha-Satellite'  # NOTE: File name issue

    make_vcf_from_read_vcf(
        sa_subfamily_vcf,
        f"COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.{sa}",
        "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
        )

    sa_subfamily_kmer_fraction = get_kmer_fraction_from_frequency_tab(get_kmer_frequency_from_interval(intervals=sa_subfamily_pr))

    get_3mer_norm_factor(
        sa_subfamily_kmer_fraction,
        df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_{sa}.tsv", sep="\t", header=None)

    sa_subfamily_kmer_fraction.columns = [f"{sa}"]

    combined_df_fraction = pd.concat([combined_df_fraction, sa_subfamily_kmer_fraction], axis=1)

combined_df_fraction.to_csv("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Kmer_Analysis/RepeatMasker/kmer3_canonical_sbs_fraction_by_repeat.tsv", sep="\t")


# %%
desired_order = ['COLO829TB_Shared_Filtered_Pruned', 'COLO829TB_Shared_Filtered_Pruned_None_RE']

# INFO: Before k-mer adjustment
raw_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned*_SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

raw_files_dataframes = []
for raw_file in raw_files:
   df = pd.read_csv(raw_file, sep='\t', index_col=0).T
   raw_files_dataframes.append(df)

raw_assignment_table = pd.concat(raw_files_dataframes, axis=1)

remaining_columns = [col for col in raw_assignment_table.columns if col not in desired_order]
final_order = desired_order + remaining_columns
raw_assignment_table = raw_assignment_table[final_order]

raw_assignment_table_fraction = raw_assignment_table.div(raw_assignment_table.sum(axis=0), axis=1)
raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction[(raw_assignment_table_fraction != 0).any(axis=1)]

# NOTE: Rename Column names
column_rename_dict = dict(zip(raw_assignment_table_fraction_nonzero.columns, list(map(lambda x: x.replace("COLO829TB_Shared_Filtered_Pruned", "").lstrip('_'), raw_assignment_table_fraction_nonzero.columns))))
column_rename_dict["COLO829TB_Shared_Filtered_Pruned"] = "All"

raw_assignment_table_fraction_nonzero = raw_assignment_table_fraction_nonzero.rename(columns=column_rename_dict)







# %%
# INFO: After k-mer adjustment
kmer_adjusted_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned*CDR_BL_50kb_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

kmer_adjusted_files.extend(sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/*Filtered_Pruned*Satellite*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt")))


kmer_adjusted_files_dataframes = []
for kmer_adjusted_file in kmer_adjusted_files:
   df = pd.read_csv(kmer_adjusted_file, sep='\t', index_col=0).T
   kmer_adjusted_files_dataframes.append(df)

kmer_adjusted_assignment_table = pd.concat(kmer_adjusted_files_dataframes, axis=1)

#remaining_columns = [col for col in kmer_adjusted_assignment_table.columns if col not in desired_order]
#final_order = desired_order + remaining_columns
#kmer_adjusted_assignment_table = kmer_adjusted_assignment_table[final_order]

kmer_adjusted_assignment_table_fraction = kmer_adjusted_assignment_table.div(kmer_adjusted_assignment_table.sum(axis=0), axis=1)
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction[(kmer_adjusted_assignment_table_fraction != 0).any(axis=1)]

column_rename_dict = {'COLO829TB_Shared_Filtered_Pruned_CDR_BL_50kb': "CDR",
                      'COLO829TB_Shared_Filtered_Pruned_NON_CDR_BL_50kb': "None-CDR",   
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_ACRO1': "ACRO1",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_BSR_Beta': "BSR_Beta",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_CER': "CER",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_GSATII': "GSATII",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_GSAT': "GSAT",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_HSat1A': "HSat1A",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_HSat2': "HSat2",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_HSat3': "HSat3",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_SATR1': "SATR1",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite': "Satellite",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_SST1': "SST1",
                      'COLO829TB_Shared_Filtered_Pruned_Satellite_alpha-Satellite': "α-Satellite"}

# NOTE: Rename Column names
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction_nonzero.rename(columns=column_rename_dict)

#distinct_sbs_satellite = set(kmer_adjusted_assignment_table_fraction_nonzero.index) - set(distinct_sbs_list)

#distinct_sbs_satellite_list = sorted(list(distinct_sbs_satellite))
#hex_satellite_colors = sns.color_palette("Set2", len(distinct_sbs_satellite_list)).as_hex()
#sbs_satellite_color_dict = dict(zip(distinct_sbs_satellite_list, hex_satellite_colors))

#sbs_color_dict.update(sbs_satellite_color_dict) # INFO: Update the existing SBS color dict with satellite colors

# INFO: Make a proportional plots
df_melted = kmer_adjusted_assignment_table_fraction_nonzero.reset_index()
df_long = df_melted.melt(id_vars=['index'], 
                         var_name='Sample_Type', 
                         value_name='Proportion')
df_long = df_long.rename(columns={'index': 'SBS_Signature'})

desired_order = ['SST1', 'SATR1', 'CER', 'GSATII', 'GSAT', 'BSR_Beta', 'ACRO1', 'HSat3', 'HSat2', 'HSat1A', 'None-CDR', 'CDR', 'α-Satellite', 'Satellite']
df_long['Sample_Type'] = pd.Categorical(df_long['Sample_Type'], categories=desired_order, ordered=True)

desired_sbs_order = ['SBS1', 'SBS5', 'SBS2', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS38', 'SBS97']
df_long['SBS_Signature'] = pd.Categorical(df_long['SBS_Signature'], categories=desired_sbs_order, ordered=True)

plot2 = (ggplot(df_long, aes(x='Sample_Type', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     # Remove the scale_x_discrete(limits=reversed) line since we're using categorical
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Proportion') +
     theme_minimal() +
     theme(
           legend_position='right',
           figure_size=(12, 8),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=14, color='black'),
           axis_title_y=element_text(size=14, color='black'),
           axis_text_x=element_text(rotation=0, size=10, color='black'),
           axis_text_y=element_text(rotation=0, size=10, color='black')
           ) +
     coord_flip()
)
ggsavefig_and_show(plot2, "Kmer-adjusted_SBS_Assignment_Satellite_subfamilies_and_CDR_vs_Non-CDR")

# %%
plot2 = (ggplot(df_long[df_long["Sample_Type"].isin(["CDR", "None-CDR"])], aes(x='Sample_Type', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     # Remove the scale_x_discrete(limits=reversed) line since we're using categorical
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Proportion') +
     theme_minimal() +
     theme(
           legend_position='right',
           figure_size=(5, 6),
           text=element_text(family='Arial'),
           plot_title=element_text(size=8, color='black'),
           axis_title_x=element_text(size=14, color='black'),
           axis_title_y=element_text(size=14, color='black'),
           axis_text_x=element_text(rotation=0, size=10, color='black'),
           axis_text_y=element_text(rotation=0, size=10, color='black')
           )
)
ggsavefig_and_show(plot2, "Kmer-adjusted_SBS_Assignment_CDR_vs_Non-CDR")

# %%
plot = (ggplot(df_long[df_long["Sample_Type"].isin(["α-Satellite", "HSat1A", "HSat2", "HSat3", "BSR_Beta"])], aes(x='Sample_Type', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     # Remove the scale_x_discrete(limits=reversed) line since we're using categorical
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Proportion') +
     theme_minimal() +
     theme(
           legend_position='right',
           figure_size=(12, 8),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=14, color='black'),
           axis_title_y=element_text(size=14, color='black'),
           axis_text_x=element_text(rotation=0, size=10, color='black'),
           axis_text_y=element_text(rotation=0, size=10, color='black')
           ) +
     coord_flip()
)     
ggsavefig_and_show(plot, "Kmer-adjusted_SBS_Assignment_Satellite_subfamilies_SNV50")

# %%
# INFO: CN4 chromosome 1 long arm SNVs (VAF ~0.5 vs VAF ~1.0)
dsa_chr1_hap1_qarm_breakpoint = pr.from_dict({
                                    'Chromosome': ['haplotype1-0000012'],
                                    'Start': [130_528_880],
                                    'End': [249_065_600]
                                    })

dsa_chr1_hap1_qarm_breakpoint = dsa_chr1_hap1_qarm_breakpoint.intersect(callable_pr)

dsa_chr1_hap1_parm_breakpoint = pr.from_dict({
                                    'Chromosome': ['haplotype1-0000012'],
                                    'Start': [0],
                                    'End': [130_528_880]
                                    })

dsa_chr1_hap1_parm_breakpoint = dsa_chr1_hap1_parm_breakpoint.intersect(callable_pr)

# %%
colotb_snvs_shared_final_filtered_pruned_chr1_hap1_parm_breakpoint_vaf = pd.DataFrame(vcf_format_getter(vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, dsa_chr1_hap1_parm_breakpoint), "VAF")).reset_index(drop=True)
colotb_snvs_shared_final_filtered_pruned_chr1_hap1_parm_breakpoint_vaf.columns = ["VAF"]
colotb_snvs_shared_final_filtered_pruned_chr1_hap1_parm_breakpoint_vaf["breakpoint"] = "p-arm-side"

colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_breakpoint_vaf = pd.DataFrame(vcf_format_getter(vcf_in_pyranges_interval(colotb_snvs_shared_final_filtered_pruned, dsa_chr1_hap1_qarm_breakpoint), "VAF")).reset_index(drop=True)
colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_breakpoint_vaf.columns = ["VAF"]
colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_breakpoint_vaf["breakpoint"] = "q-arm-side"

# NOTE: chr3_hap2 has deletion from the translocation breakpoint
colotb_snvs_shared_final_filtered_pruned_chr3_hap2_vaf = pd.DataFrame(vcf_format_getter(colotb_snvs_shared_final_filtered_pruned[colotb_snvs_shared_final_filtered_pruned['CHROM'] == 'haplotype2-0000058'], "VAF")).reset_index(drop=True)
colotb_snvs_shared_final_filtered_pruned_chr3_hap2_vaf.columns = ["VAF"]
colotb_snvs_shared_final_filtered_pruned_chr3_hap2_vaf["breakpoint"] = "chr3_hap2"


# %%
# INFO: p-arm side of chromosome 1 (upstream of translocation breakpoint)
plot = (
    ggplot(colotb_snvs_shared_final_filtered_pruned_chr1_hap1_parm_breakpoint_vaf, aes(x='VAF', fill='breakpoint')) +
    geom_histogram(breaks=np.linspace(0, 1, 51), alpha=0.7) +
    scale_fill_manual(values={'p-arm-side': 'salmon'}) +
    scale_x_continuous(limits=[0, 1]) +
    scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
    theme_minimal() +
    labs(
        title='',
        x='VAF',
        y='Count'
    ) +
    theme(figure_size=(8, 7),
          text=element_text(family='Arial'),
          strip_text=element_text(size=12, color='black'),
          axis_text=element_text(size=10, color='black'),
          axis_title=element_text(size=12, color='black'),
          plot_title=element_text(size=14, color='black'))
)
ggsavefig_and_show(plot, "vaf_histograms_by_chr1_hap1_parm_breakpoint")

# INFO: q-arm side of chromosome 1 (downstream of translocation breakpoint)
plot = (
    ggplot(colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_breakpoint_vaf, aes(x='VAF', fill='breakpoint')) +
    geom_histogram(breaks=np.linspace(0, 1, 51), alpha=0.7) +
    scale_fill_manual(values={'q-arm-side': 'magenta'}) +
    scale_x_continuous(limits=[0, 1]) +
    scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
    theme_minimal() +
    labs(
        title='',
        x='VAF',
        y='Count'
    ) +
    theme(figure_size=(8, 7),
          text=element_text(family='Arial'),
          strip_text=element_text(size=12, color='black'),
          axis_text=element_text(size=10, color='black'),
          axis_title=element_text(size=12, color='black'),
          plot_title=element_text(size=14, color='black'))
)
ggsavefig_and_show(plot, "vaf_histograms_by_chr1_hap1_qarm_breakpoint")

# INFO: chromosome 3 (SNVs available in chr3-hap2)
plot = (
    ggplot(colotb_snvs_shared_final_filtered_pruned_chr3_hap2_vaf, aes(x='VAF', fill='breakpoint')) +
    geom_histogram(breaks=np.linspace(0, 1, 51), alpha=0.7) +
    scale_fill_manual(values={'chr3_hap2': 'limegreen'}) +
    scale_x_continuous(limits=[0, 1]) +
    scale_y_continuous(expand=(0, 0, 0.05, 0), labels=comma_format()) +
    theme_minimal() +
    labs(
        title='',
        x='VAF',
        y='Count'
    ) +
    theme(figure_size=(8, 7),
          text=element_text(family='Arial'),
          strip_text=element_text(size=12, color='black'),
          axis_text=element_text(size=10, color='black'),
          axis_title=element_text(size=12, color='black'),
          plot_title=element_text(size=14, color='black'))
)
ggsavefig_and_show(plot, "vaf_histograms_by_chr3_hap2")

# %%
colotb_snvs_shared_final_filtered_pruned_chr1_qarm_0_50 = colotb_snvs_shared_final_filtered_pruned[
    (colotb_snvs_shared_final_filtered_pruned["CHROM"] == 'haplotype1-0000012') & 
    (vcf_format_getter(colotb_snvs_shared_final_filtered_pruned, "VAF") <= 0.55) & 
    (vcf_format_getter(colotb_snvs_shared_final_filtered_pruned, "VAF") >= 0.45) & 
    (colotb_snvs_shared_final_filtered_pruned["POS"] >= 130_528_880)].reset_index(drop=True)

colotb_snvs_shared_final_filtered_pruned_chr1_qarm_1_00 = colotb_snvs_shared_final_filtered_pruned[
    (colotb_snvs_shared_final_filtered_pruned["CHROM"] == 'haplotype1-0000012') & 
    (vcf_format_getter(colotb_snvs_shared_final_filtered_pruned, "VAF") >= 0.90) & 
    (colotb_snvs_shared_final_filtered_pruned["POS"] >= 130_528_880)].reset_index(drop=True)

colotb_snvs_shared_final_filtered_pruned_chr1_qarm_1_00 = colotb_snvs_shared_final_filtered_pruned[
    (colotb_snvs_shared_final_filtered_pruned["CHROM"] == 'haplotype1-0000012') & 
    (vcf_format_getter(colotb_snvs_shared_final_filtered_pruned, "VAF") >= 0.90) & 
    (colotb_snvs_shared_final_filtered_pruned["POS"] >= 130_528_880)].reset_index(drop=True)

outdir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"

make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_chr1_qarm_0_50,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.chr1_hap1_qarm_VAF_0.5",
    outdir
)

make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_chr1_qarm_1_00,
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.chr1_hap1_qarm_VAF_1.0",
    outdir
)

dsa_chr1_hap1_qarm_breakpoint_kmer_fraction = get_kmer_fraction_from_frequency_tab(
    get_kmer_frequency_from_interval(intervals=dsa_chr1_hap1_qarm_breakpoint))

get_3mer_norm_factor(
    dsa_chr1_hap1_qarm_breakpoint_kmer_fraction,
    df_canonical_sbs96_fraction
    ).to_csv(f"{outdir}/kmer3_norm_factor_dsa_chr1_hap1_qarm_breakpoint.tsv", sep="\t", header=None)

# %%
# INFO: After k-mer adjustment
kmer_adjusted_files = sorted(glob("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_*_SBS96/Kmer_normalization/Assignment_Solution/Activities/Assignment_Solution_Activities.txt"))

kmer_adjusted_files_dataframes = []
for kmer_adjusted_file in kmer_adjusted_files:
   df = pd.read_csv(kmer_adjusted_file, sep='\t', index_col=0).T
   kmer_adjusted_files_dataframes.append(df)

kmer_adjusted_assignment_table = pd.concat(kmer_adjusted_files_dataframes, axis=1)

kmer_adjusted_assignment_table_fraction = kmer_adjusted_assignment_table.div(kmer_adjusted_assignment_table.sum(axis=0), axis=1)
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction[(kmer_adjusted_assignment_table_fraction != 0).any(axis=1)]

column_rename_dict = {'COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5': "VAF~0.5",
                      'COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0': "VAF~1.0"}

# NOTE: Rename Column names
kmer_adjusted_assignment_table_fraction_nonzero = kmer_adjusted_assignment_table_fraction_nonzero.rename(columns=column_rename_dict)

# INFO: Make a proportional plots
df_melted = kmer_adjusted_assignment_table_fraction_nonzero.reset_index()
df_long = df_melted.melt(id_vars=['index'], 
                         var_name='Sample_Type', 
                         value_name='Proportion')
df_long = df_long.rename(columns={'index': 'SBS_Signature'})

desired_order = ['VAF~1.0', 'VAF~0.5']
df_long['Sample_Type'] = pd.Categorical(df_long['Sample_Type'], categories=desired_order, ordered=True)

desired_order = ['SBS1', 'SBS7a', 'SBS7b', 'SBS38', 'SBS97']
df_long['SBS_Signature'] = pd.Categorical(df_long['SBS_Signature'], categories=desired_order, ordered=True)

plot2 = (ggplot(df_long, aes(x='Sample_Type', y='Proportion', fill='SBS_Signature')) +
     geom_col(position='stack', width=0.8) +
     scale_y_continuous(labels=lambda x: [f'{v:.1%}' for v in x]) +
     scale_fill_manual(values=sbs_color_dict, name='SBS Signature') +
     labs(title='Proportion of SBS Signatures between different genomic contexts\n(3-mer Post-Normalization)',
          x='Genomic Context',
          y='Proportion') +
     theme_minimal() +
     theme(
           legend_position='right',
           figure_size=(5, 5),
           text=element_text(family='Arial'),
           plot_title=element_text(size=10, color='black'),
           axis_title_x=element_text(size=10, color='black'),
           axis_title_y=element_text(size=10, color='black'),
           axis_text_x=element_text(rotation=0, size=8, color='black'),
           axis_text_y=element_text(rotation=0, size=8, color='black')
           )
)
ggsavefig_and_show(plot2, "Kmer-adjusted_SBS_Assignment_chr1_hap1_qarm_vaf0.5_vs_vaf1.0")
# %%
# INFO: SBS96 for chr1 q-arm SNVs (VAF ~0.5 vs VAF ~1.0):
# INFO: VAF ~1.0
sbs6_colors = {'C>A': '#03bcee',
          'C>G': 'black',
          'C>T': '#e32926',
          'T>A': '#cac9c9',
          'T>C': '#a1ce63',
          'T>G': '#ebc6c4'}

sbs6_order = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

vaf_1_0_sbs_matrix=pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0_SBS96/Kmer_normalization/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0.SBS96.kmer_normalized.all", sep="\t")
vaf_1_0_sbs_matrix['sbs6'] = vaf_1_0_sbs_matrix['MutationType'].str.extract(r'\[([ACGT]>[ACGT])\]')

vaf_1_0_sbs_matrix['sbs6'] = pd.Categorical(
                        vaf_1_0_sbs_matrix['sbs6'], 
                        categories=sbs6_order, 
                        ordered=True)

vaf_1_0_sbs_matrix = vaf_1_0_sbs_matrix.sort_values(['sbs6', 'MutationType']).reset_index(drop=True)

vaf_1_0_sbs_matrix['MutationType'] = pd.Categorical(
    vaf_1_0_sbs_matrix['MutationType'], 
    categories=vaf_1_0_sbs_matrix['MutationType'].tolist(),
    ordered=True)

vaf_1_0_sbs_matrix["Percentage"] = vaf_1_0_sbs_matrix['COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0']*100 / vaf_1_0_sbs_matrix['COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0'].sum()

plot = (ggplot(vaf_1_0_sbs_matrix, aes(x='MutationType', y='COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_1_0', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Count') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_vaf1.0_kmer_normalized")

plot = (ggplot(vaf_1_0_sbs_matrix, aes(x='MutationType', y='Percentage', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Percentage') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_vaf1.0_kmer_normalized_percentage")

# INFO: VAF ~0.5
sbs6_colors = {'C>A': '#03bcee',
          'C>G': 'black',
          'C>T': '#e32926',
          'T>A': '#cac9c9',
          'T>C': '#a1ce63',
          'T>G': '#ebc6c4'}

sbs6_order = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

vaf_0_5_sbs_matrix=pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5_SBS96/Kmer_normalization/COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5.SBS96.kmer_normalized.all", sep="\t")
vaf_0_5_sbs_matrix['sbs6'] = vaf_0_5_sbs_matrix['MutationType'].str.extract(r'\[([ACGT]>[ACGT])\]')

vaf_0_5_sbs_matrix['sbs6'] = pd.Categorical(
                        vaf_0_5_sbs_matrix['sbs6'], 
                        categories=sbs6_order, 
                        ordered=True)

vaf_0_5_sbs_matrix = vaf_0_5_sbs_matrix.sort_values(['sbs6', 'MutationType']).reset_index(drop=True)

vaf_0_5_sbs_matrix['MutationType'] = pd.Categorical(
    vaf_0_5_sbs_matrix['MutationType'], 
    categories=vaf_0_5_sbs_matrix['MutationType'].tolist(),
    ordered=True)

vaf_0_5_sbs_matrix["Percentage"] = vaf_0_5_sbs_matrix['COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5']*100 / vaf_0_5_sbs_matrix['COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5'].sum()

plot = (ggplot(vaf_0_5_sbs_matrix, aes(x='MutationType', y='COLO829TB_Shared_Filtered_Pruned_chr1_breakpoint_VAF_0_5', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Count') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_vaf0.5_kmer_normalized")

plot = (ggplot(vaf_0_5_sbs_matrix, aes(x='MutationType', y='Percentage', fill='sbs6')) +
        geom_col() +
        scale_fill_manual(values=sbs6_colors) +
        labs(title='SBS96 Mutation Spectrum',
             x='Mutation Type',
             y='Percentage') +
        theme_tufte() +
        theme(figure_size=(8, 2.5),
              text=element_text(family='Arial'),
              axis_text_x=element_text(color='black', rotation=90, hjust=0.5, size=5),
              axis_text_y=element_text(color='black'),
              plot_title=element_text(size=12, color='black', hjust=0.5),
              axis_line_x=element_line(size=0.5, color='black'),
              axis_line_y=element_line(size=0.5, color='black'),
              axis_ticks_major=element_line(size=0.2, color='black'),
              axis_ticks_minor_x=element_line(size=0.02, color='darkgray'),
              axis_ticks_minor_y=element_line(size=0.02, color='darkgray'))
        )
ggsavefig_and_show(plot, "sbs96_spectrum_colotb_snvs_shared_final_filtered_pruned_chr1_hap1_qarm_vaf0.5_kmer_normalized_percentage")

# %%
# INFO: Probability of getting breakpoints in CDR by chance for i(4)(q10) and t(14;16)(p10;p10)
# NOTE: "Centromere" is defined by identifying the segments in DSA corresponds to the T2T-CM13 centromere regions
# NOTE: for i(4)(q10), chr4-hap1 (centromere -> haplotype1-0000013:133652173-146572350;length: 12_395_840)
# NOTE: for i(4)(q10), breakpoints in CDR is in hap2 (centromere -> haplotype2-0000060:133531402-146553424;length: 13_022_022)
# NOTE: for t(14;16)(p10;p10),  chr16-hap1 (centromere -> haplotype1-0000008:12352608-37650615; length: 7_248_279)
# NOTE: for t(14;16)(p10;p10), breakpoints in CDR is in chr14-hap2 (centromere -> haplotype2-0000070:3758-18425397;length: 18_421_639)

i4q10_chr4_hap1_pr = pr.from_dict({
    'Chromosome': ["haplotype1-0000013"],
    'Start': [133_652_173],
    'End': [146_572_350]
})

i4q10_chr4_hap1_pr = i4q10_chr4_hap1_pr.intersect(callable_pr)

print(
    cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype1-0000013"].length / i4q10_chr4_hap1_pr.length
)

i4q10_chr4_hap2_pr = pr.from_dict({
    'Chromosome': ["haplotype2-0000060"],
    'Start': [133_531_402],
    'End': [146_553_424]
})

i4q10_chr4_hap2_pr = i4q10_chr4_hap2_pr.intersect(callable_pr)

print(
    cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype2-0000060"].length / i4q10_chr4_hap2_pr.length
)

t14_16_p10_p10_chr16_hap1_pr = pr.from_dict({
    'Chromosome': ["haplotype1-0000008"],
    'Start': [12_352_608],
    'End': [37_650_615]
})

t14_16_p10_p10_chr16_hap1_pr = t14_16_p10_p10_chr16_hap1_pr.intersect(callable_pr)

print(
    cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype1-0000008"].length / t14_16_p10_p10_chr16_hap1_pr.length
)

t14_16_p10_p10_chr14_hap2_pr = pr.from_dict({
    'Chromosome': ["haplotype2-0000070"],
    'Start': [3758],
    'End': [18_425_397]
})

t14_16_p10_p10_chr14_hap2_pr = t14_16_p10_p10_chr14_hap2_pr.intersect(callable_pr)

print(
    cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype2-0000070"].length / t14_16_p10_p10_chr14_hap2_pr.length
)

# %%
def simulate_process(a1, c1, a2, c2, a3, c3, a4, c4, num=100_000):
    """
    IF (Random number from 1 to a1) ≤ c1 then 1 else 0
    IF (Random number from 1 to a2) ≤ c2 then 1 else 0
    IF (Random number from 1 to a3) ≤ c3 then 1 else 0
    IF (Random number from 1 to a4) ≤ c4 then 1 else 0

    IF SUM(Outcome of above 4 IF statements) ≥ 2 then 1 else 0

    Repeat the above for 10,000 times and count how many times the final outcome is 1 = P-value
    """
    success_count = 0
    random.seed(1024) # NOTE: Reproducibility
    for _ in range(num):
        outcome1 = 1 if random.randint(1, c1) <= a1 else 0
        outcome2 = 1 if random.randint(1, c2) <= a2 else 0
        outcome3 = 1 if random.randint(1, c3) <= a3 else 0
        outcome4 = 1 if random.randint(1, c4) <= a4 else 0
        
        total = outcome1 + outcome2 + outcome3 + outcome4

        final_outcome = 1 if total >= 2 else 0
        
        if final_outcome == 1:
            success_count += 1
    
    p_value = success_count / num
    
    return p_value, success_count

a1, c1 = cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype1-0000013"].length, i4q10_chr4_hap1_pr.length
a2, c2 = cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype2-0000060"].length, i4q10_chr4_hap2_pr.length
a3, c3 = cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype1-0000008"].length, t14_16_p10_p10_chr16_hap1_pr.length
a4, c4 = cdr_bl_50kb_extended_pr[cdr_bl_50kb_extended_pr.Chromosome == "haplotype2-0000070"].length, t14_16_p10_p10_chr14_hap2_pr.length

p_value, successes = simulate_process(a1, c1, a2, c2, a3, c3, a4, c4)

print(f"Results after 100,000 simulations:")
print(f"P-value: {p_value:.4f}")
print(f"P-value as percentage: {p_value*100:.2f}%")


# %%
# INFO: Pentanucleotide context 
# NOTE: for example, TC[C>T]AT

test = get_kmer_frequency_from_interval(intervals=sa_pr[sa_pr.Name4 == "HSat2"].merge(), k=5)
test_fraction = test_fraction = test.div(test.sum(axis=0), axis=1).reset_index()


# %%
test2 = get_kmer_frequency_from_interval(intervals=sa_pr[sa_pr.Name4 == "HSat2"].merge(), k=7)
test2_fraction = test2.div(test2.sum(axis=0), axis=1).reset_index()


# %%
# INFO: Mutational index after accounting for 3-mer context
kmer3_enrichment_vector = combined_df_fraction.div(combined_df_fraction["None_RE"], axis=0)

kmer3_adjusted_percentage = dict()
kmer3_adjusted_percentage_over_percentage = dict()

re_of_interest_dict.update({
    'CDR': 'COLO829TB_Shared_Filtered_Pruned_CDR_BL_50kb',
    'Non-CDR': 'COLO829TB_Shared_Filtered_Pruned_NON_CDR_BL_50kb'
})

for j, k in re_of_interest_dict.items():

    sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/{k}_SBS96/Kmer_normalization/{k}.SBS96.kmer_normalized.all", sep="\t")
    sbs_matrix["Percentage"] = sbs_matrix[f'{k}']*100 / sbs_matrix[f'{k}'].sum()
 
    sbs_matrix["3mer"] = sbs_matrix["MutationType"].apply(lambda x: x[0] + x[2] + x[6])

    if j == 'α-Satellite':
        re_index = 'alpha-Satellite'  # NOTE: File name issue

    else:
        re_index = j

    enrichment_vector_of_interest = kmer3_enrichment_vector[re_index].to_dict()

    sbs_matrix['Adjusted_Percentage'] = sbs_matrix['Percentage'] * sbs_matrix['3mer'].map(enrichment_vector_of_interest)

    #print(f"{re_index}: {sbs_matrix['Percentage'].sum()} (Should be very close to 100.0)")
    print(f"{re_index}: {sbs_matrix['Adjusted_Percentage'].sum() * 0.01}")

    kmer3_adjusted_percentage[re_index] = sbs_matrix['Adjusted_Percentage'].sum()
    kmer3_adjusted_percentage_over_percentage[re_index] = sbs_matrix['Adjusted_Percentage'].sum() * 0.01

# %%
snvrate_nonere = rc_length_snv_count_df_filtered_plot[rc_length_snv_count_df_filtered_plot["Category"] == "None_RE"]["SNV_rate"].values

test1 = pd.DataFrame.from_dict(kmer3_adjusted_percentage_over_percentage, orient='index')
test1 = test1[test1.index.isin(['alpha-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3'])]
test1 = test1.rename(index={"alpha-Satellite": "α-Satellite"}).reset_index()
test1.columns = ["Category", "Mutability"]

test2 = rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot[rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot["Category"].isin(['α-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3'])][["Category", "SNV_rate"]].reset_index(drop=True)

test3 = pd.merge(test1, test2, on="Category")
test3["Enrichment_over_noneRE_rate"] = test3["SNV_rate"] / snvrate_nonere

slope, intercept, r, p, _ = linregress(test3['Enrichment_over_noneRE_rate'], test3['Mutability'])
r_squared = r**2

# %%
x_pos = test3['Enrichment_over_noneRE_rate'].max() * 0.7
y_pos = test3['Mutability'].max() * 0.4

plot = (ggplot(test3, aes(x='Enrichment_over_noneRE_rate', y='Mutability')) +
        geom_point(size=3, alpha=0.7) +
        geom_smooth(method='lm', se=False, linetype='dashed', color='red', size=0.5) +
        scale_x_continuous(trans='log10', breaks=[1, 2, 3, 5, 8], limits=[0.99, 8]) +
        scale_y_continuous(trans='log10') +
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        annotate('text', x=x_pos, y=y_pos, 
                 label=f'R² = {r_squared:.3f}\np = {p:.4f}', 
                 size=8, 
                 ha='left') +
        labs(title='',
             x='Enrichment over None_RE Mutational Rate',
             y='Mutability') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "mutational_rate_enrichment_vs_mutability_satellite_subfamilies")

plot = (ggplot(test3, aes(x='Enrichment_over_noneRE_rate', y='Mutability')) +
        geom_point(size=3, alpha=0.7) +
        geom_smooth(method='lm', se=True, linetype='dashed', color='red', size=0.5) +
        scale_x_continuous(trans='log10', breaks=[1, 2, 3, 5, 8], limits=[0.99, 8]) +
        scale_y_continuous(trans='log10') +
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        annotate('text', x=x_pos, y=y_pos, 
                 label=f'R² = {r_squared:.3f}\np = {p:.4f}', 
                 size=8, 
                 ha='left') +
        labs(title='',
             x='Enrichment over None_RE Mutational Rate',
             y='Mutability') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "mutational_rate_enrichment_vs_mutability_satellite_subfamilies_with_confidence_interval")

plot = (ggplot(test3, aes(x='Enrichment_over_noneRE_rate', y='Mutability')) +
        geom_point(size=3, alpha=0.7) +
        geom_smooth(method='lm', se=True, linetype='dashed', color='red', size=0.5) +
        scale_x_continuous(limits=[0, 8]) +
        scale_y_continuous(limits=[0, 8]) +        
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        annotate('text', x=x_pos, y=y_pos, 
                 label=f'R² = {r_squared:.3f}\np = {p:.4f}', 
                 size=8, 
                 ha='left') +
        labs(title='',
             x='Enrichment over None_RE Mutational Rate',
             y='Mutability') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "mutational_rate_enrichment_vs_mutability_satellite_subfamilies_with_confidence_interval_nonlog")


# %%
test1 = pd.DataFrame.from_dict(kmer3_adjusted_percentage_over_percentage, orient='index')
test1_2 = test1[test1.index.isin(['alpha-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3', 'CDR', 'Non-CDR'])]
test1_2 = test1_2.rename(index={"alpha-Satellite": "α-Satellite"}).reset_index()
test1_2.columns = ["Category", "Mutability"]

test2_2 = rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot[rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot["Category"].isin(['α-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3'])][["Category", "SNV_rate"]].reset_index(drop=True)
test4_2 = pd.DataFrame({"Category": ["CDR", "Non-CDR"], "SNV_rate": [colotb_snvs_shared_final_filtered_pruned_cdr_bl_50kb.shape[0] / cdr_bl_50kb_extended_pr.length, colotb_snvs_shared_final_filtered_pruned_non_cdr_bl_50kb.shape[0] / non_cdr_bl_50kb_extended_pr.length]}) # NOTE: Add CDR and Non-CDR SNV rate
test2_2 = pd.concat([test2_2, test4_2])

test3_2 = pd.merge(test1_2, test2_2, on="Category")
test3_2["Enrichment_over_noneRE_rate"] = test3_2["SNV_rate"] / snvrate_nonere

# NOTE: Draw regression line
plot = (ggplot(test3_2, aes(x='Enrichment_over_noneRE_rate', y='Mutability')) +
        geom_point(size=3, alpha=0.7) +
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        scale_x_continuous(trans='log10', breaks=[1, 2, 3, 5, 8], limits=[0.99, 8]) +
        scale_y_continuous(trans='log10') +
        labs(title='',
             x='Enrichment over None_RE Mutational Rate',
             y='Mutability') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "mutational_rate_enrichment_vs_mutability_satellite_subfamilies_and_cdr")


# %%
test1 = pd.DataFrame.from_dict(kmer3_adjusted_percentage_over_percentage, orient='index')
test1_3 = test1[~test1.index.isin(['alpha-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3'])].reset_index()
test1_3.columns = ["Category", "Mutability"]

test2_3 = rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot[~rc_length_snv_count_df_filtered_plot_with_sa_wo_total_plot["Category"].isin(['α-Satellite', 'HSat2', 'HSat1A', 'BSR_Beta', 'HSat3'])][["Category", "SNV_rate"]].reset_index(drop=True)

test3_3 = pd.merge(test1_3, test2_3, on="Category")
test3_3["Enrichment_over_noneRE_rate"] = test3_3["SNV_rate"] / snvrate_nonere

# NOTE: Draw regression line
plot = (ggplot(test3_3, aes(x='Enrichment_over_noneRE_rate', y='Mutability')) +
        geom_point(size=3, alpha=0.7) +
        geom_text(aes(label='Category'), 
                  adjust_text=adjust_text_dict,
                  size=4) +
        scale_x_continuous(trans='log10', breaks=[0.1, 1, 2, 3, 5], limits=[0.9, 5]) +
        scale_y_continuous(trans='log10', breaks=[0.7, 0.8, 0.9, 1, 1.5], limits=[0.8, 1.5]) +
        labs(title='',
             x='Enrichment over None_RE Mutational Rate',
             y='Mutability') +
        theme_minimal() +
        theme(
           figure_size=(3, 3),
           text=element_text(family='Arial'),
           plot_title=element_text(size=14, color='black'),
           axis_title_x=element_text(size=7, color='black'),
           axis_title_y=element_text(size=7, color='black'),
           axis_text_x=element_text(rotation=0, size=6, color='black'),
           axis_text_y=element_text(rotation=0, size=6, color='black'),
           legend_position='none'
           )
        )

ggsavefig_and_show(plot, "mutational_rate_enrichment_vs_mutability_all_repeat_element")

# %%
# INFO: TFBS and Genome-wide Mutability

# INFO: TFBS 3-mer fraction
tf_kmer_fraction = pd.read_table("/mmfs1/gscratch/stergachislab/sjn/projects/per-tf-kmer/results.kmer/results.mtx", sep="\t", index_col=0).T
tf_kmer_fraction = tf_kmer_fraction.iloc[1:]

index_rename_dict = dict()

for k in tf_kmer_fraction.index:
    # NOTE: Only applied to odd-length kmers
    middle_base = k[len(k) // 2]
    if middle_base not in ["C", "T"]:
        index_rename_dict[k] = reverse_complement(k)
    else:
        index_rename_dict[k] = k

tf_kmer_fraction = tf_kmer_fraction.rename(index=index_rename_dict)
tf_kmer_fraction = tf_kmer_fraction.reindex(kmer3_enrichment_vector.index)
tf_kmer_fraction = tf_kmer_fraction.div(tf_kmer_fraction.sum(axis=0), axis=1)

# INFO: Genome-wide (GRCh38) 3-mer fraction
hg38_genomewide_kmer_fraction = pd.read_table("/mmfs1/gscratch/stergachislab/mvollger/projects/software/rustybam/all-hg38-3-mers.tbl", sep="\t", index_col=0).T
hg38_genomewide_kmer_fraction = hg38_genomewide_kmer_fraction.iloc[1:]
hg38_genomewide_kmer_fraction = hg38_genomewide_kmer_fraction.rename(index=index_rename_dict)
hg38_genomewide_kmer_fraction = hg38_genomewide_kmer_fraction.reindex(kmer3_enrichment_vector.index)
hg38_genomewide_kmer_fraction = hg38_genomewide_kmer_fraction.div(hg38_genomewide_kmer_fraction.sum(axis=0), axis=1)

# INFO: TFBS 3-mer enrichment vector (over GRCh38 Genome-wide 3-mer fraction)
kmer3_enrichment_vector_for_tf_hg38 = tf_kmer_fraction.div(hg38_genomewide_kmer_fraction["All"], axis=0)


colo829_sbs_matrix = pd.read_table(f"/mmfs1/gscratch/stergachislab/mhsohny/Tools/VCF2SPECTRUM/results/COLO829TB_Shared_Filtered_Pruned_SBS96/Kmer_normalization/COLO829TB_Shared_Filtered_Pruned.SBS96.kmer_normalized.all", sep="\t")
colo829_sbs_matrix["Percentage"] = colo829_sbs_matrix["COLO829TB_Shared_Filtered_Pruned"]*100 / colo829_sbs_matrix["COLO829TB_Shared_Filtered_Pruned"].sum()

colo829_sbs_matrix["3mer"] = colo829_sbs_matrix["MutationType"].apply(lambda x: x[0] + x[2] + x[6])

tf_hg38_kmer3_adjusted_percentage = dict()

for i in kmer3_enrichment_vector_for_tf_hg38:

    enrichment_vector_of_interest = kmer3_enrichment_vector_for_tf_hg38[i].to_dict()

    tf_hg38_kmer3_adjusted_percentage[i] = (colo829_sbs_matrix['Percentage'] * colo829_sbs_matrix['3mer'].map(enrichment_vector_of_interest)).values

tf_hg38_kmer3_adjusted_percentage_df = pd.DataFrame(tf_hg38_kmer3_adjusted_percentage, index=colo829_sbs_matrix["MutationType"])

tf_hg38_kmer3_adjusted_percentage_sum_df = pd.DataFrame((tf_hg38_kmer3_adjusted_percentage_df.sum(axis=0) * 0.01).sort_values(ascending=True), columns=["Mutability"])

tf_hg38_kmer3_adjusted_percentage_sum_df_plot = tf_hg38_kmer3_adjusted_percentage_sum_df.reset_index()
tf_hg38_kmer3_adjusted_percentage_sum_df_plot.columns = ["TF", "Mutability"]
tf_hg38_kmer3_adjusted_percentage_sum_df_plot['Position'] = range(len(tf_hg38_kmer3_adjusted_percentage_sum_df_plot))

tf_hg38_kmer3_adjusted_percentage_sum_df_plot["TF_name"] = tf_hg38_kmer3_adjusted_percentage_sum_df_plot["TF"].str.split('_').str[-1]

tf_hg38_kmer3_adjusted_percentage_sum_df_plot['TF_plot'] = ''
tf_hg38_kmer3_adjusted_percentage_sum_df_plot.loc[tf_hg38_kmer3_adjusted_percentage_sum_df_plot.index[-30:], 'TF_plot'] = tf_hg38_kmer3_adjusted_percentage_sum_df_plot.loc[
    tf_hg38_kmer3_adjusted_percentage_sum_df_plot.index[-30:], 'TF'
].str.split('_').str[-1]

adjust_text_dict = {
    'expand': (10, 4.5),
    'arrowprops': {
        'arrowstyle': '-'
        }
    }

plot = (
    ggplot(tf_hg38_kmer3_adjusted_percentage_sum_df_plot, aes(x='Position', y='Mutability')) +
    geom_point(position=position_jitter(width=0.001, height=0), 
               alpha=0.8, size=1, stroke=0.1, color='black', fill="none") +
    geom_hline(yintercept=1, color='red', linetype='solid') +
    scale_y_continuous(trans='log10', breaks=[0, 0.5, 1, 2, 3]) +
    labs(x='', y='Predicted Mutatbility') +
    theme_minimal() +
    theme(
        figure_size=(2, 2),
        text=element_text(family='Arial'),
        axis_text_x=element_blank(),
        axis_title_y=element_text(size=7, color='black'),
        axis_text_y=element_text(rotation=0, size=6, color='black')
        )
)

ggsavefig_and_show(plot, "tfbs_genomewide_mutability")
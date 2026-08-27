# %%
#%%HTML
#<style>
#    body {
#        --vscode-font-family: "CaskaydiaCove Nerd Font"
#    }
#</style>

# %%
import io
import os
import csv
import numpy as np
import pandas as pd
from itertools import chain
import pyranges as pr
import gzip as gz
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib_venn import venn2, venn3
from supervenn import supervenn # type: ignore
from plotnine import *
from mizani.formatters import scientific_format, comma_format
import sigProfilerPlotting as sigPlt
from SigProfilerAssignment import Analyzer as Analyze
from scipy.stats import fisher_exact
from scipy.stats import binomtest
from statsmodels.stats import multitest
from pathlib import Path
from tqdm import tqdm
import subprocess
from pyfaidx import Fasta

sns.set_theme(font="Arial", font_scale=1.15, style='ticks') 
matplotlib.rcParams['figure.dpi'] = 150
plt.rc("axes.spines", top=False, right=False)
#%matplotlib inline

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
    width=6.4,
    height=4.8,
):
    """
    plot: plotnine.ggplot.ggplot
    """
    plot.save(f"{plotdir}/{filename}.pdf", dpi=dpi, width=width, height=height)

    os.system(f"code -r {plotdir}/{filename}.pdf")

def snv_pr_metrics(called: set, truth: set, extra_fn: int = 0):
    """TP/FP/FN + precision/recall for a called SNV set vs a truth set.
    extra_fn adds un-recallable truth items (e.g. DSA SNVs not surjectable to the
    reference) to the recall_withreject denominator; extra_fn=0 -> == recall."""
    tp = len(called & truth)
    fp = len(called - truth)
    fn = len(truth - called)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    recall_withreject = tp / (tp + fn + extra_fn)
    return tp, fp, fn, precision, recall, recall_withreject

def reverse_complement(string):
    try:
        complement_dict = {"A": "T", "T": "A", "G": "C", "C": "G"}
        complement_string = "".join([complement_dict[s.upper()] for s in string])
    except KeyError:
        raise ValueError("Invalid character other than A,T,G and C")
    return complement_string[::-1]

def read_vcf(path):
    if path[-3:] == ".gz": 
        with gz.open(path, 'rb') as f:
            lines = [l.decode('utf-8') for l in f if not l.startswith(b'##')]
            return pd.read_csv(
                io.StringIO(''.join(lines)),
                dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
                       'QUAL': str, 'FILTER': str, 'INFO': str},
                       sep='\t'
                       ).rename(columns={'#CHROM': 'CHROM'})
    else:
        with open(path, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]
            return pd.read_csv(
                io.StringIO(''.join(lines)),
                dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
                       'QUAL': str, 'FILTER': str, 'INFO': str},
                       sep='\t'
                       ).rename(columns={'#CHROM': 'CHROM'})

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
    format = list(set(df['FORMAT'].values))[0].split(':')

    if field == "GT":
        gtindex = format.index('GT')
        return df[sampleid].str.split(':').apply(lambda x: x[gtindex])
    
    elif field == "VAF":
        vafindex = format.index('VAF')
        return df[sampleid].str.split(':').apply(lambda x: float(x[vafindex]))
    
    elif field == "DP":
        dpindex = format.index('DP')
        return df[sampleid].str.split(':').apply(lambda x: int(x[dpindex]))
    
    elif field == "AD":
        adindex = format.index('AD')
        return df[sampleid].str.split(':').apply(lambda x: int(x[adindex].split(',')[1]))
    
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
    if info_string == "." or not info_string:
        return {}
    
    result = {}
    for i in info_string.split(';'):
        if '=' in i:  # Make sure there's an equals sign to split on
            key, value = i.split('=', 1)  # Split only on first occurrence of =
            result[key] = value
    return result

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
    def safe_get_field(info_str):
        parsed = vcf_info_parser(info_str)
        if not parsed or field not in parsed:
            return np.nan
        
        value = parsed[field]
        if value == 'NA':
            return np.nan
            
        if field in ("VAF_Ill", "VAF_PB"):
            try:
                return float(value)
            except (ValueError, TypeError):
                return np.nan
        return value
    
    return df['INFO'].apply(safe_get_field)
    
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
    
    df = df[['CHROM', 'POS', 'POS']].drop_duplicates()
    
    df.to_csv(f"{os.path.join(path, prefix)}.sitelist", sep='\t', index=False, header=False)


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

def make_vcf_from_read_vcf(df: pd.DataFrame, template: str, prefix: str, outdir: str) -> None:

    # /mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz

    '''
    Parameters
    ---
    df : pandas.core.frame.DataFrame
        VCF read through read_vcf()
    template : str
        VCF of which header will be used as the template for the output VCF
    prefix : str
    outdir : str

    Usage Example
    ---
    make_vcf_from_read_vcf(
    colotb_snvs_shared_final_filtered_pruned_cdr_bl_100kb,
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/VariantCalls_DeepVariant_1.6.1/COLO829T_PassageB_DSA/deepvariant/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.vcf.gz",
    "COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.CDR-BL_100kb",
    "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering"
    )
    '''

    # INFO: Generate Pseudo-VCF Header
    os.system(
        f"zcat '{template}' \
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

    # INFO: Remove Pseudo VCF Header
    os.system(
        f"rm {outdir}/pseudovcf_header"
        )

dsa="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0.fasta"
hg38="/mmfs1/gscratch/stergachislab/assemblies/hg38.analysisSet.fa"
chm13="/mmfs1/gscratch/stergachislab/assemblies/chm13v2.0.fa"

dsa_pyfaidx = Fasta(dsa, rebuild=False)
hg38_pyfaidx = Fasta(hg38, rebuild=False)
chm13_pyfaidx = Fasta(chm13, rebuild=False)

def get_fasta_sequence(reference: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    """
    0-based BED file needed
    """
    if strand == "+":
        return reference[chrom][start:end].seq.upper()
    elif strand == "-":
        return reference[chrom][start:end].complement.seq.upper()

def get_sequence_from_bed(df: pd.DataFrame, reference: Fasta) -> pd.DataFrame:
    sequences = []
    for _, row in df.iterrows():
        if row["Strand"] == '+':
            seq = get_fasta_sequence(reference, row['Chromosome'], row['Start'], row['End'], row["Strand"])
            sequences.append(seq)
        elif row["Strand"] == "-":
            seq = get_fasta_sequence(reference, row['Chromosome'], row['Start'], row['End'], row["Strand"])
            sequences.append(seq)
    df['Seq'] = sequences
    return df

# %%
# INFO: DSA-only Callable Regions (after removing 100kb-DEL and Flagger-NucFlag regions) vs. GRCh38
dsa_only_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toGRCh38/COLO829BL_hap1_2_unaligned_to_GRCh38_Flagger-NucFlag_removed.bed.gz", header=None, sep="\t")

dsa_only_pr = pr.from_dict({
    'Chromosome': dsa_only_bed.iloc[:, 0],
    'Start': dsa_only_bed.iloc[:, 1],
    'End': dsa_only_bed.iloc[:, 2],
})

dsa_only_pr.Length = dsa_only_pr.lengths()
dsa_only_pr_1kb = dsa_only_pr[dsa_only_pr.df["Length"] >= 1000].copy()
#dsa_only_pr_500bp = dsa_only_pr[dsa_only_pr.df["Length"] >= 500].copy()

# %%
# INFO: DSA-only Callable Regions (after removing 100kb-DEL and Flagger-NucFlag regions) vs. T2T-CHM13
dsa_only_chm13_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toCHM13/COLO829BL_hap1_2_unaligned_to_CHM13_Flagger-NucFlag_removed.bed.gz", header=None, sep="\t")

dsa_only_chm13_pr = pr.from_dict({
    'Chromosome': dsa_only_chm13_bed.iloc[:, 0],
    'Start': dsa_only_chm13_bed.iloc[:, 1],
    'End': dsa_only_chm13_bed.iloc[:, 2],
})

dsa_only_chm13_pr.Length = dsa_only_chm13_pr.lengths()
dsa_only_chm13_pr_1kb = dsa_only_chm13_pr[dsa_only_chm13_pr.df["Length"] >= 1000].copy()
#dsa_only_chm13_pr_500bp = dsa_only_chm13_pr[dsa_only_chm13_pr.df["Length"] >= 500].copy()

# %%
# INFO: Callable Regions of the DSA (Removing Flagger-NucFlag and Deleted segments of the genome)
callable_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_Flagger-NucFlag_100kb-DEL_removed.bed.gz", header=None, sep="\t")
callable_pr = pr.from_dict({
    'Chromosome': callable_bed.iloc[:, 0],
    'Start': callable_bed.iloc[:, 1],
    'End': callable_bed.iloc[:, 2]
})

# %% 
# INFO: DSA-based Reference SNV-set
colotb_shared_snv = read_vcf("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.vcf.gz")
colotb_shared_snv['SNVid'] = colotb_shared_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
colotb_shared_snv['POSid'] = colotb_shared_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
snv_referenceset = set(colotb_shared_snv['SNVid'].values) # NOTE: 44,795 # NOTE: This will be reduced to 44,366 after removing the overlapped SNVs with the overlapped_leftover_snv below
#snv_referenceset_position_set = set(colotb_shared_snv['POSid'].values)

snv_referenceset_pgfbsnvid_snvid_tab = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered.dict.tsv", names=['pgfbSNVid', 'SNVid'])

pgfbsnvid_to_snvid = dict(zip(snv_referenceset_pgfbsnvid_snvid_tab['pgfbSNVid'], snv_referenceset_pgfbsnvid_snvid_tab['SNVid']))
snvid_to_pgfbsnvid= dict(zip(snv_referenceset_pgfbsnvid_snvid_tab['SNVid'], snv_referenceset_pgfbsnvid_snvid_tab['pgfbSNVid']))

overlapped_leftover_snv = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/Overlapped_leftover_SNV.tsv", sep="\t")
overlapped_leftover_snv_list = list(chain(*list(overlapped_leftover_snv.values))) # NOTE: 462
overlapped_leftover_snv_list_position_set = set(map(lambda x: '_'.join(x.split('_')[:2]), overlapped_leftover_snv_list)) # NOTE: 456

snv_referenceset = snv_referenceset.difference(set(overlapped_leftover_snv_list)) # NOTE: 44,366 (Not directly translated to 44,795 - 462 = 44,333 because some SNVs in the overlapped_leftover_snv_list are not in the snv_referenceset upon SNV Density Filtering step)
snv_referenceset_position_set = set(map(lambda x: '_'.join(x.split('_')[:2]), snv_referenceset))

# %%
# INFO: Flagged SNVid (DSA)
snv_flagset_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV/SNV_Flagset.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"]) # NOTE: Caution here. pgfbSNVid is not the same as DSA-based Reference SNV-set pgfbSNVid.

snv_flagset_position_set = set(snv_flagset_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)
snv_flagset_position_set = snv_flagset_position_set.union(overlapped_leftover_snv_list_position_set) # NOTE: 35,811

# %%
# %% [markdown]
# DSA-based Reference SNV-set injected to DSG and surject into the other haplotype (hap1->hap2 and vice versa)
# ADDED: For short-read (Illumina), in the DSA, due to randomly assigning of reads to one of the two haplotypes when there is homology between two haplotypes, precision of the SNV calls in the DSA-space is under-estimated. Therefore, we need to take into account of this by using -- also -- the SNV calls from the other haplotype.
# ADDED: Hap1<->Hap2 Surjected DSA-based SNV Reference set
surject_between_hap_dir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/SCNA_Adjusted/SNV_Density-based_Filtering/Surject_to_other_hap"
dsa_hap2_to_hap1 = pd.read_table(f"{surject_between_hap_dir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_COLO829BL_hap1_withTags_noSecondaryFlag_sorted.bed", header=None, names=["Chromosome", "Start", "End", "pgfbSNVid", "HapOfOrigin", "Strand"])
dsa_hap1_to_hap2 = pd.read_table(f"{surject_between_hap_dir}/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_COLO829BL_hap2_withTags_noSecondaryFlag_sorted.bed", header=None, names=["Chromosome", "Start", "End", "pgfbSNVid", "HapOfOrigin", "Strand"])

# NOTE: Mapping pgfbSNVID
# NOTE: SNV in originally in hap2 surjected to hap1
dsa_hap2_to_hap1_originhap2 = dsa_hap2_to_hap1[dsa_hap2_to_hap1["HapOfOrigin"] == 2].merge(snv_referenceset_pgfbsnvid_snvid_tab, on="pgfbSNVid", how="left")

dsa_hap2_to_hap1_originhap2_ref_alt = dsa_hap2_to_hap1_originhap2["SNVid"].str.rsplit("_", n=2, expand=True)

dsa_hap2_to_hap1_originhap2_ref, dsa_hap2_to_hap1_originhap2_alt = dsa_hap2_to_hap1_originhap2_ref_alt[1], dsa_hap2_to_hap1_originhap2_ref_alt[2]

dsa_hap2_to_hap1_originhap2_reverse = dsa_hap2_to_hap1_originhap2["Strand"] == "-"

dsa_hap2_to_hap1_originhap2_ref = dsa_hap2_to_hap1_originhap2_ref.where(~dsa_hap2_to_hap1_originhap2_reverse, dsa_hap2_to_hap1_originhap2_ref.map(reverse_complement))
dsa_hap2_to_hap1_originhap2_alt = dsa_hap2_to_hap1_originhap2_alt.where(~dsa_hap2_to_hap1_originhap2_reverse, dsa_hap2_to_hap1_originhap2_alt.map(reverse_complement))

dsa_hap2_to_hap1_originhap2_snvid = dsa_hap2_to_hap1_originhap2["Chromosome"] + "_" + dsa_hap2_to_hap1_originhap2["End"].astype(str) + "_" + dsa_hap2_to_hap1_originhap2_ref + "_" + dsa_hap2_to_hap1_originhap2_alt

# NOTE: SNV in originally in hap1 surjected to hap2
dsa_hap1_to_hap2_originhap1 = dsa_hap1_to_hap2[dsa_hap1_to_hap2["HapOfOrigin"] == 1].merge(snv_referenceset_pgfbsnvid_snvid_tab, on="pgfbSNVid", how="left")

dsa_hap1_to_hap2_originhap1_ref_alt = dsa_hap1_to_hap2_originhap1["SNVid"].str.rsplit("_", n=2, expand=True)

dsa_hap1_to_hap2_originhap1_ref, dsa_hap1_to_hap2_originhap1_alt = dsa_hap1_to_hap2_originhap1_ref_alt[1], dsa_hap1_to_hap2_originhap1_ref_alt[2]

dsa_hap1_to_hap2_originhap1_reverse = dsa_hap1_to_hap2_originhap1["Strand"] == "-"

dsa_hap1_to_hap2_originhap1_ref = dsa_hap1_to_hap2_originhap1_ref.where(~dsa_hap1_to_hap2_originhap1_reverse, dsa_hap1_to_hap2_originhap1_ref.map(reverse_complement))
dsa_hap1_to_hap2_originhap1_alt = dsa_hap1_to_hap2_originhap1_alt.where(~dsa_hap1_to_hap2_originhap1_reverse, dsa_hap1_to_hap2_originhap1_alt.map(reverse_complement))

dsa_hap1_to_hap2_originhap1_snvid = dsa_hap1_to_hap2_originhap1["Chromosome"] + "_" + dsa_hap1_to_hap2_originhap1["End"].astype(str) + "_" + dsa_hap1_to_hap2_originhap1_ref + "_" + dsa_hap1_to_hap2_originhap1_alt

snv_referenceset_between_hap_surjected = snv_referenceset | ((set(dsa_hap2_to_hap1_originhap2_snvid.values) | set(dsa_hap1_to_hap2_originhap1_snvid.values)) - set(overlapped_leftover_snv_list))


# %%
# ADDED: Making Non-Satellite Regions of the DSA
# INFO: DSA Satellite Regions
dsa_satellite_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/Rhodonite/RepeatMasker/RM_DSA_COLO829BL_v3.0.0_Satellite_Callable_COLO829.bed", header=None, sep="\t")
dsa_satellite_pr = pr.from_dict({
    'Chromosome': dsa_satellite_bed.iloc[:, 0],
    'Start': dsa_satellite_bed.iloc[:, 1],
    'End': dsa_satellite_bed.iloc[:, 2]
})

dsa_callable_non_satellite_pr = callable_pr.subtract(dsa_satellite_pr)

# INFO: MAKING DSA-based Reference set of non-satellite callable regions
colotb_shared_snv_non_satellite = vcf_in_pyranges_interval(colotb_shared_snv, dsa_callable_non_satellite_pr)
snv_referenceset_non_satellite = set(colotb_shared_snv_non_satellite['SNVid'].values)

snv_referenceset_non_satellite = snv_referenceset_non_satellite.difference(set(overlapped_leftover_snv_list))
snv_referenceset_non_satellite_position_set = set(map(lambda x: '_'.join(x.split('_')[:2]), snv_referenceset_non_satellite))


# INFO: GRCh38 Satellite Regions
hg38_satellite_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/References/GRCh38/hg38_Satellite.bed", header=None, sep="\t")
hg38_satellite_pr = pr.from_dict({
    'Chromosome': hg38_satellite_bed.iloc[:, 0],
    'Start': hg38_satellite_bed.iloc[:, 1],
    'End': hg38_satellite_bed.iloc[:, 2]
})

# INFO: T2T-CHM13
chm13_satellite_bed = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/Tools/References/T2T-CHM13v2.0/chm13v2.0_Satellite.bed", header=None, sep="\t")
chm13_satellite_pr = pr.from_dict({
    'Chromosome': chm13_satellite_bed.iloc[:, 0],
    'Start': chm13_satellite_bed.iloc[:, 1],
    'End': chm13_satellite_bed.iloc[:, 2]
})


# %%
# INFO: DeepSomatic Tumor-only-mode Somatic SNVs
insilico_dsa_dir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/in_silico_mixture/DSA"

bl_100x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_100x_snv['SNVid'] = bl_100x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_snv = bl_100x_snv[(vcf_info_getter(bl_100x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_100x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_100x_snv = vcf_in_pyranges_interval(bl_100x_snv, callable_pr)

# %%
## INFO: DeepSomatic Tumor-Normal-Pair-mode Somatic SNVs 
## NOTE: Not used in the current study
#deepsomatic_dir="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Fiber-seq/VariantCalls_DeepSomatic_1.8.0"
#colotb_deepsomatictonly_snv = read_vcf(f"{deepsomatic_dir}/COLO829T_PassageB_DSA/COLO829T_PassageB_DSA.deepsomatictonly.PASS.snv.annot.FlaggerHap.NucFlag.vcf.gz")
#colotb_deepsomatictonly_snv['SNVid'] = colotb_deepsomatictonly_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

# %%
# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures
blt_t1n4_100x = read_vcf(f"{insilico_dsa_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_100x = read_vcf(f"{insilico_dsa_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_100x = read_vcf(f"{insilico_dsa_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_100x = blt_t1n4_100x[(vcf_info_getter(blt_t1n4_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_100x = blt_t1n9_100x[(vcf_info_getter(blt_t1n9_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_100x = blt_t1n19_100x[(vcf_info_getter(blt_t1n19_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_100x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_100x['SNVid'] = blt_t1n4_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_100x['SNVid'] = blt_t1n9_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_100x['SNVid'] = blt_t1n19_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_100x["POSid"] = blt_t1n4_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_100x["POSid"] = blt_t1n9_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_100x["POSid"] = blt_t1n19_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_100x = blt_t1n4_100x[~blt_t1n4_100x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n9_100x = blt_t1n9_100x[~blt_t1n9_100x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n19_100x = blt_t1n19_100x[~blt_t1n19_100x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)

blt_t1n4_100x = vcf_in_pyranges_interval(blt_t1n4_100x, callable_pr)
blt_t1n9_100x = vcf_in_pyranges_interval(blt_t1n9_100x, callable_pr)
blt_t1n19_100x = vcf_in_pyranges_interval(blt_t1n19_100x, callable_pr)

blt_t1n4_10x = read_vcf(f"{insilico_dsa_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_20x = read_vcf(f"{insilico_dsa_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_40x = read_vcf(f"{insilico_dsa_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_10x = blt_t1n4_10x[(vcf_info_getter(blt_t1n4_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_20x = blt_t1n4_20x[(vcf_info_getter(blt_t1n4_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_40x = blt_t1n4_40x[(vcf_info_getter(blt_t1n4_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_10x['SNVid'] = blt_t1n4_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_20x['SNVid'] = blt_t1n4_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_40x['SNVid'] = blt_t1n4_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_10x["POSid"] = blt_t1n4_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_20x["POSid"] = blt_t1n4_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_40x["POSid"] = blt_t1n4_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_10x = blt_t1n4_10x[~blt_t1n4_10x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n4_20x = blt_t1n4_20x[~blt_t1n4_20x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n4_40x = blt_t1n4_40x[~blt_t1n4_40x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)

blt_t1n4_10x = vcf_in_pyranges_interval(blt_t1n4_10x, callable_pr)
blt_t1n4_20x = vcf_in_pyranges_interval(blt_t1n4_20x, callable_pr)
blt_t1n4_40x = vcf_in_pyranges_interval(blt_t1n4_40x, callable_pr)

blt_t1n9_10x = read_vcf(f"{insilico_dsa_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_20x = read_vcf(f"{insilico_dsa_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_40x = read_vcf(f"{insilico_dsa_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n9_10x = blt_t1n9_10x[(vcf_info_getter(blt_t1n9_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_20x = blt_t1n9_20x[(vcf_info_getter(blt_t1n9_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_40x = blt_t1n9_40x[(vcf_info_getter(blt_t1n9_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n9_10x['SNVid'] = blt_t1n9_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_20x['SNVid'] = blt_t1n9_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_40x['SNVid'] = blt_t1n9_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n9_10x["POSid"] = blt_t1n9_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_20x["POSid"] = blt_t1n9_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_40x["POSid"] = blt_t1n9_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n9_10x = blt_t1n9_10x[~blt_t1n9_10x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n9_20x = blt_t1n9_20x[~blt_t1n9_20x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n9_40x = blt_t1n9_40x[~blt_t1n9_40x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)

blt_t1n9_10x = vcf_in_pyranges_interval(blt_t1n9_10x, callable_pr)
blt_t1n9_20x = vcf_in_pyranges_interval(blt_t1n9_20x, callable_pr)
blt_t1n9_40x = vcf_in_pyranges_interval(blt_t1n9_40x, callable_pr)

blt_t1n19_10x = read_vcf(f"{insilico_dsa_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_20x = read_vcf(f"{insilico_dsa_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_40x = read_vcf(f"{insilico_dsa_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n19_10x = blt_t1n19_10x[(vcf_info_getter(blt_t1n19_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_20x = blt_t1n19_20x[(vcf_info_getter(blt_t1n19_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_40x = blt_t1n19_40x[(vcf_info_getter(blt_t1n19_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n19_10x['SNVid'] = blt_t1n19_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_20x['SNVid'] = blt_t1n19_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_40x['SNVid'] = blt_t1n19_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n19_10x["POSid"] = blt_t1n19_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_20x["POSid"] = blt_t1n19_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_40x["POSid"] = blt_t1n19_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n19_10x = blt_t1n19_10x[~blt_t1n19_10x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n19_20x = blt_t1n19_20x[~blt_t1n19_20x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n19_40x = blt_t1n19_40x[~blt_t1n19_40x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)

blt_t1n19_10x = vcf_in_pyranges_interval(blt_t1n19_10x, callable_pr)
blt_t1n19_20x = vcf_in_pyranges_interval(blt_t1n19_20x, callable_pr)
blt_t1n19_40x = vcf_in_pyranges_interval(blt_t1n19_40x, callable_pr)

# %%
# INFO: Calculating Precision and Recall in the DSA space
# INFO: T1N4 100X
blt_t1n4_100x_snv = set(blt_t1n4_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_100x_snv_tp, blt_t1n4_100x_snv_fp, blt_t1n4_100x_snv_fn, blt_t1n4_100x_snv_precision, blt_t1n4_100x_snv_recall, _ = snv_pr_metrics(blt_t1n4_100x_snv, snv_referenceset)


# INFO: T1N4 40X
blt_t1n4_40x_snv = set(blt_t1n4_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_40x_snv_tp, blt_t1n4_40x_snv_fp, blt_t1n4_40x_snv_fn, blt_t1n4_40x_snv_precision, blt_t1n4_40x_snv_recall, _ = snv_pr_metrics(blt_t1n4_40x_snv, snv_referenceset)


# INFO: T1N4 20X
blt_t1n4_20x_snv = set(blt_t1n4_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_20x_snv_tp, blt_t1n4_20x_snv_fp, blt_t1n4_20x_snv_fn, blt_t1n4_20x_snv_precision, blt_t1n4_20x_snv_recall, _ = snv_pr_metrics(blt_t1n4_20x_snv, snv_referenceset)


# INFO: T1N4 10X
blt_t1n4_10x_snv = set(blt_t1n4_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_10x_snv_tp, blt_t1n4_10x_snv_fp, blt_t1n4_10x_snv_fn, blt_t1n4_10x_snv_precision, blt_t1n4_10x_snv_recall, _ = snv_pr_metrics(blt_t1n4_10x_snv, snv_referenceset)


# INFO: T1N9 100X
blt_t1n9_100x_snv = set(blt_t1n9_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_100x_snv_tp, blt_t1n9_100x_snv_fp, blt_t1n9_100x_snv_fn, blt_t1n9_100x_snv_precision, blt_t1n9_100x_snv_recall, _ = snv_pr_metrics(blt_t1n9_100x_snv, snv_referenceset)


# INFO: T1N9 40X
blt_t1n9_40x_snv = set(blt_t1n9_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_40x_snv_tp, blt_t1n9_40x_snv_fp, blt_t1n9_40x_snv_fn, blt_t1n9_40x_snv_precision, blt_t1n9_40x_snv_recall, _ = snv_pr_metrics(blt_t1n9_40x_snv, snv_referenceset)


# INFO: T1N9 20X
blt_t1n9_20x_snv = set(blt_t1n9_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n9_20x_snv_tp, blt_t1n9_20x_snv_fp, blt_t1n9_20x_snv_fn, blt_t1n9_20x_snv_precision, blt_t1n9_20x_snv_recall, _ = snv_pr_metrics(blt_t1n9_20x_snv, snv_referenceset)


# INFO: T1N9 10X
blt_t1n9_10x_snv = set(blt_t1n9_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_10x_snv_tp, blt_t1n9_10x_snv_fp, blt_t1n9_10x_snv_fn, blt_t1n9_10x_snv_precision, blt_t1n9_10x_snv_recall, _ = snv_pr_metrics(blt_t1n9_10x_snv, snv_referenceset)


# INFO: T1N19 100X
blt_t1n19_100x_snv = set(blt_t1n19_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n19_100x_snv_tp, blt_t1n19_100x_snv_fp, blt_t1n19_100x_snv_fn, blt_t1n19_100x_snv_precision, blt_t1n19_100x_snv_recall, _ = snv_pr_metrics(blt_t1n19_100x_snv, snv_referenceset)


# INFO: T1N19 40X
blt_t1n19_40x_snv = set(blt_t1n19_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_40x_snv_tp, blt_t1n19_40x_snv_fp, blt_t1n19_40x_snv_fn, blt_t1n19_40x_snv_precision, blt_t1n19_40x_snv_recall, _ = snv_pr_metrics(blt_t1n19_40x_snv, snv_referenceset)


# INFO: T1N19 20X
blt_t1n19_20x_snv = set(blt_t1n19_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_20x_snv_tp, blt_t1n19_20x_snv_fp, blt_t1n19_20x_snv_fn, blt_t1n19_20x_snv_precision, blt_t1n19_20x_snv_recall, _ = snv_pr_metrics(blt_t1n19_20x_snv, snv_referenceset)


# INFO: T1N19 10X
blt_t1n19_10x_snv = set(blt_t1n19_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_10x_snv_tp, blt_t1n19_10x_snv_fp, blt_t1n19_10x_snv_fn, blt_t1n19_10x_snv_precision, blt_t1n19_10x_snv_recall, _ = snv_pr_metrics(blt_t1n19_10x_snv, snv_referenceset)

# %%
# ADDED: Non-Satellite Regions of the DSA
bl_100x_snv_non_satellite = vcf_in_pyranges_interval(bl_100x_snv, dsa_callable_non_satellite_pr)

blt_t1n4_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_20x, dsa_callable_non_satellite_pr)
blt_t1n9_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_20x, dsa_callable_non_satellite_pr)
blt_t1n19_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_20x, dsa_callable_non_satellite_pr)


# INFO: T1N4
blt_t1n4_20x_non_satellite_snv = set(blt_t1n4_20x_non_satellite["SNVid"].values) - set(bl_100x_snv_non_satellite["SNVid"].values) 
blt_t1n4_20x_non_satellite_snv_tp, blt_t1n4_20x_non_satellite_snv_fp, blt_t1n4_20x_non_satellite_snv_fn, blt_t1n4_20x_non_satellite_snv_precision, blt_t1n4_20x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n4_20x_non_satellite_snv, snv_referenceset_non_satellite)

# INFO: T1N9
blt_t1n9_20x_non_satellite_snv = set(blt_t1n9_20x_non_satellite["SNVid"].values) - set(bl_100x_snv_non_satellite["SNVid"].values) 
blt_t1n9_20x_non_satellite_snv_tp, blt_t1n9_20x_non_satellite_snv_fp, blt_t1n9_20x_non_satellite_snv_fn, blt_t1n9_20x_non_satellite_snv_precision, blt_t1n9_20x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n9_20x_non_satellite_snv, snv_referenceset_non_satellite)

# INFO: T1N19
blt_t1n19_20x_non_satellite_snv = set(blt_t1n19_20x_non_satellite["SNVid"].values) - set(bl_100x_snv_non_satellite["SNVid"].values) 
blt_t1n19_20x_non_satellite_snv_tp, blt_t1n19_20x_non_satellite_snv_fp, blt_t1n19_20x_non_satellite_snv_fn, blt_t1n19_20x_non_satellite_snv_precision, blt_t1n19_20x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n19_20x_non_satellite_snv, snv_referenceset_non_satellite)


# %%
# ADDED: Illumina short-read (Diploid 100X, haploid 50X) for evaluating mSNV discovery performance
# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures 
platform="Illumina"
bl_sr_50x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/{platform}/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_sr_50x_snv['SNVid'] = bl_sr_50x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_sr_50x_snv = bl_sr_50x_snv[(vcf_info_getter(bl_sr_50x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_sr_50x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_sr_50x_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, callable_pr)

blt_t1n4_sr_50x = read_vcf(f"{insilico_dsa_dir}/T1N4/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_sr_50x = read_vcf(f"{insilico_dsa_dir}/T1N9/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_sr_50x = read_vcf(f"{insilico_dsa_dir}/T1N19/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_sr_50x = blt_t1n4_sr_50x[(vcf_info_getter(blt_t1n4_sr_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_sr_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_sr_50x = blt_t1n9_sr_50x[(vcf_info_getter(blt_t1n9_sr_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_sr_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_sr_50x = blt_t1n19_sr_50x[(vcf_info_getter(blt_t1n19_sr_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_sr_50x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_sr_50x['SNVid'] = blt_t1n4_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_50x['SNVid'] = blt_t1n9_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_50x['SNVid'] = blt_t1n19_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_50x["POSid"] = blt_t1n4_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_50x["POSid"] = blt_t1n9_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_50x["POSid"] = blt_t1n19_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_50x = blt_t1n4_sr_50x[~blt_t1n4_sr_50x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n9_sr_50x = blt_t1n9_sr_50x[~blt_t1n9_sr_50x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)
blt_t1n19_sr_50x = blt_t1n19_sr_50x[~blt_t1n19_sr_50x["POSid"].isin(snv_flagset_position_set)].reset_index(drop=True)

blt_t1n4_sr_50x = vcf_in_pyranges_interval(blt_t1n4_sr_50x, callable_pr)
blt_t1n9_sr_50x = vcf_in_pyranges_interval(blt_t1n9_sr_50x, callable_pr)
blt_t1n19_sr_50x = vcf_in_pyranges_interval(blt_t1n19_sr_50x, callable_pr)

# %%
# INFO: Calculating Precision and Recall in the DSA space but FOR SHORT-READ (Illumina) sequencing data
# INFO: T1N4 50X
blt_t1n4_sr_50x_snv = set(blt_t1n4_sr_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n4_sr_50x_snv_tp, blt_t1n4_sr_50x_snv_fp, blt_t1n4_sr_50x_snv_fn, blt_t1n4_sr_50x_snv_precision, blt_t1n4_sr_50x_snv_recall, _ = snv_pr_metrics(blt_t1n4_sr_50x_snv, snv_referenceset)


# INFO: T1N9 50X
blt_t1n9_sr_50x_snv = set(blt_t1n9_sr_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n9_sr_50x_snv_tp, blt_t1n9_sr_50x_snv_fp, blt_t1n9_sr_50x_snv_fn, blt_t1n9_sr_50x_snv_precision, blt_t1n9_sr_50x_snv_recall, _ = snv_pr_metrics(blt_t1n9_sr_50x_snv, snv_referenceset)


# INFO: T1N19 50X
blt_t1n19_sr_50x_snv = set(blt_t1n19_sr_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n19_sr_50x_snv_tp, blt_t1n19_sr_50x_snv_fp, blt_t1n19_sr_50x_snv_fn, blt_t1n19_sr_50x_snv_precision, blt_t1n19_sr_50x_snv_recall, _ = snv_pr_metrics(blt_t1n19_sr_50x_snv, snv_referenceset)

# %%
# INFO: Using `snv_referenceset_between_hap_surjected` to calculate Precision
# INFO: T1N4 50X
blt_t1n4_sr_50x_snv_hap_surject_tp, blt_t1n4_sr_50x_snv_hap_surject_fp, blt_t1n4_sr_50x_snv_hap_surject_fn, blt_t1n4_sr_50x_snv_hap_surject_precision, blt_t1n4_sr_50x_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n4_sr_50x_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N9 50X
blt_t1n9_sr_50x_snv_hap_surject_tp, blt_t1n9_sr_50x_snv_hap_surject_fp, blt_t1n9_sr_50x_snv_hap_surject_fn, blt_t1n9_sr_50x_snv_hap_surject_precision, blt_t1n9_sr_50x_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n9_sr_50x_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N19 50X
blt_t1n19_sr_50x_snv_hap_surject_tp, blt_t1n19_sr_50x_snv_hap_surject_fp, blt_t1n19_sr_50x_snv_hap_surject_fn, blt_t1n19_sr_50x_snv_hap_surject_precision, blt_t1n19_sr_50x_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n19_sr_50x_snv, snv_referenceset_between_hap_surjected)

# %%
# INFO: Restricted to Satellite Region (DSA)
bl_sr_50x_non_satellite_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, dsa_callable_non_satellite_pr)

blt_t1n4_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_sr_50x, dsa_callable_non_satellite_pr)
blt_t1n9_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_sr_50x, dsa_callable_non_satellite_pr)
blt_t1n19_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_sr_50x, dsa_callable_non_satellite_pr)


# INFO: Calculating Precision and Recall in the DSA space but FOR SHORT-READ (Illumina) sequencing data but in Non-Satellite Region Only
# INFO: T1N4 50X
blt_t1n4_sr_50x_non_satellite_snv = set(blt_t1n4_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_50x_non_satellite_snv["SNVid"].values)
blt_t1n4_sr_50x_non_satellite_snv_tp, blt_t1n4_sr_50x_non_satellite_snv_fp, blt_t1n4_sr_50x_non_satellite_snv_fn, blt_t1n4_sr_50x_non_satellite_snv_precision, blt_t1n4_sr_50x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n4_sr_50x_non_satellite_snv, snv_referenceset_non_satellite)


# INFO: T1N9 50X
blt_t1n9_sr_50x_non_satellite_snv = set(blt_t1n9_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_50x_non_satellite_snv["SNVid"].values)
blt_t1n9_sr_50x_non_satellite_snv_tp, blt_t1n9_sr_50x_non_satellite_snv_fp, blt_t1n9_sr_50x_non_satellite_snv_fn, blt_t1n9_sr_50x_non_satellite_snv_precision, blt_t1n9_sr_50x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n9_sr_50x_non_satellite_snv, snv_referenceset_non_satellite)


# INFO: T1N19 50X
blt_t1n19_sr_50x_non_satellite_snv = set(blt_t1n19_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_50x_non_satellite_snv["SNVid"].values)
blt_t1n19_sr_50x_non_satellite_snv_tp, blt_t1n19_sr_50x_non_satellite_snv_fp, blt_t1n19_sr_50x_non_satellite_snv_fn, blt_t1n19_sr_50x_non_satellite_snv_precision, blt_t1n19_sr_50x_non_satellite_snv_recall, _ = snv_pr_metrics(blt_t1n19_sr_50x_non_satellite_snv, snv_referenceset_non_satellite)

# INFO: Using `snv_referenceset_between_hap_surjected` to calculate Precision but in Non-Satellite Region Only
# INFO: T1N4 50X
blt_t1n4_sr_50x_non_satellite_snv_hap_surject_tp, blt_t1n4_sr_50x_non_satellite_snv_hap_surject_fp, blt_t1n4_sr_50x_non_satellite_snv_hap_surject_fn, blt_t1n4_sr_50x_non_satellite_snv_hap_surject_precision, blt_t1n4_sr_50x_non_satellite_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n4_sr_50x_non_satellite_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N9 50X
blt_t1n9_sr_50x_non_satellite_snv_hap_surject_tp, blt_t1n9_sr_50x_non_satellite_snv_hap_surject_fp, blt_t1n9_sr_50x_non_satellite_snv_hap_surject_fn, blt_t1n9_sr_50x_non_satellite_snv_hap_surject_precision, blt_t1n9_sr_50x_non_satellite_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n9_sr_50x_non_satellite_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N19 50X
blt_t1n19_sr_50x_non_satellite_snv_hap_surject_tp, blt_t1n19_sr_50x_non_satellite_snv_hap_surject_fp, blt_t1n19_sr_50x_non_satellite_snv_hap_surject_fn, blt_t1n19_sr_50x_non_satellite_snv_hap_surject_precision, blt_t1n19_sr_50x_non_satellite_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n19_sr_50x_non_satellite_snv, snv_referenceset_between_hap_surjected)



# %%
pr_data = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_100x_snv_precision, blt_t1n4_40x_snv_precision, blt_t1n4_20x_snv_precision, blt_t1n4_10x_snv_precision,
                         blt_t1n9_100x_snv_precision, blt_t1n9_40x_snv_precision, blt_t1n9_20x_snv_precision, blt_t1n9_10x_snv_precision,
                         blt_t1n19_100x_snv_precision, blt_t1n19_40x_snv_precision, blt_t1n19_20x_snv_precision, blt_t1n19_10x_snv_precision],
           "Recall": [blt_t1n4_100x_snv_recall, blt_t1n4_40x_snv_recall, blt_t1n4_20x_snv_recall, blt_t1n4_10x_snv_recall,
                      blt_t1n9_100x_snv_recall, blt_t1n9_40x_snv_recall, blt_t1n9_20x_snv_recall, blt_t1n9_10x_snv_recall,
                      blt_t1n19_100x_snv_recall, blt_t1n19_40x_snv_recall, blt_t1n19_20x_snv_recall, blt_t1n19_10x_snv_recall]}

pr_df = pd.DataFrame(pr_data)
# %%
pr_df['Coverage_numeric'] = pr_df['Coverage'].str.replace('X', '').astype(int)

ordered_ratios = ['T1N4', 'T1N9', 'T1N19']
category_colors = sns.color_palette("Paired", 9)
pr_df['Ratio'] = pd.Categorical(pr_df['Ratio'], categories=ordered_ratios, ordered=True)

pr_df = pr_df.sort_values(by=['Ratio', 'Coverage_numeric'], ascending=[True, False])

pr_long = pd.melt(pr_df, 
                  id_vars=['Ratio', 'Coverage', 'Coverage_numeric'], 
                  value_vars=['Precision', 'Recall'],
                  var_name='Metric', 
                  value_name='Value')

ratio_mapping = {
    'T1N4': '1:4 Tumor:Normal mixture\n(~20% VAF)',
    'T1N9': '1:9 Tumor:Normal mixture\n(~10% VAF)', 
    'T1N19': '1:19 Tumor:Normal mixture\n(~5% VAF)'
}

pr_long['Ratio_Label'] = pr_long['Ratio'].map(ratio_mapping)

coverage_order = ['100X', '40X', '20X', '10X']
pr_long['Coverage'] = pd.Categorical(pr_long['Coverage'], categories=coverage_order, ordered=True)

plot = (ggplot(pr_long, aes(x='Coverage', y='Value', group='Ratio')) +
        geom_line(size=1, color='darkred') +
        geom_point(size=2, color='darkred') +
        facet_grid('Metric ~ Ratio_Label', scales='free_y') +  # Use Ratio_Label instead of Ratio
        scale_y_continuous(breaks=np.arange(0, 1.25, 0.25), 
                          labels=['0.0', '0.25', '0.5', '0.75', '1.0'],
                          limits=(0, 1)) +
        labs(title='Precision and Recall by Coverage and Tumor:Normal Ratio (DSA-only)',
             x='Sequencing Coverage',
             y='') +
        theme_minimal() +
        theme(
              text=element_text(family='Arial'),
              axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
              axis_text_y=element_text(color='black'),
              axis_title_x=element_text(color='black'),
              axis_title_y=element_text(color='black'),
              plot_title=element_text(color='black'),
              strip_text=element_text(size=10, face='bold', color='black')))

ggsavefig_and_show(plot, "Precision_and_Recall_by_Coverage_and_Ratio_DSA-only")


# %% [markdown]
# DSA-based Reference SNV-set injected to DSG and surjected to GRCh38

# %%
# INFO: DSA-based Reference SNV-set injected to DSG and surjected to GRCh38
snv_referenceset_hg38_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toGRCh38/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_GRCh38_withTags_sorted.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_referenceset_hg38_position = get_sequence_from_bed(snv_referenceset_hg38_position, hg38_pyfaidx)
snv_referenceset_hg38_position["SNVid"] = snv_referenceset_hg38_position['pgfbSNVid'].map(pgfbsnvid_to_snvid)
# DEBUG: snv_referenceset_hg38_position[~(snv_referenceset_hg38_position["Seq"] == snv_referenceset_hg38_position['SNVid'].str.split('_').str[-2])]

snv_referenceset_hg38_position["SNV_hg38_base"] = snv_referenceset_hg38_position["SNVid"].str.split("_").str[-1]

snv_referenceset_hg38_position['SNVid_hg38'] = np.where(
    snv_referenceset_hg38_position['Strand'] == '+',
    snv_referenceset_hg38_position['Chromosome'] + '_' + snv_referenceset_hg38_position['End'].astype(str) + '_' + snv_referenceset_hg38_position['Seq'] + '_' + snv_referenceset_hg38_position['SNV_hg38_base'],
    snv_referenceset_hg38_position['Chromosome'] + '_' + snv_referenceset_hg38_position['End'].astype(str) + '_' + snv_referenceset_hg38_position['Seq'].apply(reverse_complement) + '_' + snv_referenceset_hg38_position['SNV_hg38_base'].apply(reverse_complement)
) # NOTE: it's a little mess for historical reason, but it works

#snv_referenceset_hg38_position_set = set(snv_referenceset_hg38_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)

snv_referenceset_hg38_position_set = set(snv_referenceset_hg38_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)
snv_referenceset_hg38_set = set(snv_referenceset_hg38_position["SNVid_hg38"].values)

# NOTE: DeepVariant/DeepSomatic only uses primary chromosomes. See https://github.com/google/deepvariant/blob/6fc7e0fc7edda9f84fdcb2e6d2ab965602a729c8/deepvariant/exclude_contigs.py#L4
primary_chrs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'] 

snv_referenceset_hg38_position_primary = snv_referenceset_hg38_position[snv_referenceset_hg38_position["Chromosome"].isin(primary_chrs)].reset_index(drop=True)
snv_referenceset_hg38_position_primary_set = set(snv_referenceset_hg38_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values) # NOTE: This was previously used for filtering (Deprecated and replaced by snv_referenceset_hg38_primary_set)

# INFO: Flagged SNV sites (DSA-based Flagged Set + overlapped SNVs-set surject-injected to GRCh38)
snv_flagset_hg38_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV/SNV_Flagset_GRCh38_peaks__surj_onto_GRCh38_withTags_sorted.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_flagset_hg38_position_set = set(snv_flagset_hg38_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values) # NOTE: 15,348

overlapped_leftover_snv_list_df = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV/Overlapped_leftover_SNV_GRCh38_peaks__surj_onto_GRCh38_withTags_sorted.bed", sep="\t", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_flagset_hg38_position_set = snv_flagset_hg38_position_set.union(set(overlapped_leftover_snv_list_df[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values))

# NOTE: Subtracting Flagged SNV set from GRCh38-surjected Reference (Primary) SNV set
condition1 = snv_referenceset_hg38_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).isin(snv_flagset_hg38_position_set)
condition2 = snv_referenceset_hg38_position_primary['SNVid'].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").isin(overlapped_leftover_snv_list_position_set) # NOTE: May seem redundant but necessary to ensure all overlapped SNVs are excluded from the DSA-based Reference Set

snv_referenceset_hg38_primary_set = set(snv_referenceset_hg38_position_primary[~condition1 & ~condition2]['SNVid_hg38'].values) # NOTE: 32,882

# ADDED: Non-satellite SNV set for GRCh38
snv_referenceset_hg38_position_primary_pr = pr.from_dict({
    'Chromosome': snv_referenceset_hg38_position_primary["Chromosome"],
    'Start': snv_referenceset_hg38_position_primary["Start"],
    'End': snv_referenceset_hg38_position_primary["End"]
})

snv_referenceset_hg38_position_primary_non_satellite_pr = snv_referenceset_hg38_position_primary_pr.subtract(hg38_satellite_pr)
snv_referenceset_hg38_position_primary_non_satellite_position_set = set(snv_referenceset_hg38_position_primary_non_satellite_pr.df[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)

condition3 = snv_referenceset_hg38_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).isin(snv_referenceset_hg38_position_primary_non_satellite_position_set)

snv_referenceset_hg38_primary_non_satellite_set = set(snv_referenceset_hg38_position_primary[~condition1 & ~condition2 & condition3]['SNVid_hg38'].values) # NOTE:  32,806

# ADDED: Non-satellite regions for GRCh38 
hg38_fai = pd.read_table("/mmfs1/gscratch/stergachislab/assemblies/hg38.analysisSet.fa.fai", header=None, sep="\t", names=["Chromosome", "End", "_1", "_2", "_3"])
hg38_fai.insert(1, "Start", 0)
hg38_bed = hg38_fai[hg38_fai["Chromosome"].isin(primary_chrs)][["Chromosome", "Start", "End"]]

hg38_pr = pr.from_dict({
    'Chromosome': hg38_bed["Chromosome"],
    'Start': hg38_bed["Start"],
    'End': hg38_bed["End"]
})

hg38_non_satellite_pr = hg38_pr.subtract(hg38_satellite_pr)

# %% [markdown]
# DSA-based Reference SNV-set injected to DSG and surjected to T2T-CHM13

# %%
# INFO: DSA-based Reference SNV-set injected to DSG and surjected to T2T-CHM13
snv_referenceset_chm13_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toCHM13/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_CHM13_withTags_sorted.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_referenceset_chm13_position = get_sequence_from_bed(snv_referenceset_chm13_position, chm13_pyfaidx)
snv_referenceset_chm13_position["SNVid"] = snv_referenceset_chm13_position['pgfbSNVid'].map(pgfbsnvid_to_snvid)
# DEBUG: snv_referenceset_chm13_position[~(snv_referenceset_chm13_position["Seq"] == snv_referenceset_chm13_position['SNVid'].str.split('_').str[-2])]

snv_referenceset_chm13_position["SNV_chm13_base"] = snv_referenceset_chm13_position["SNVid"].str.split("_").str[-1]

snv_referenceset_chm13_position['SNVid_chm13'] = np.where(
    snv_referenceset_chm13_position['Strand'] == '+',
    snv_referenceset_chm13_position['Chromosome'] + '_' + snv_referenceset_chm13_position['End'].astype(str) + '_' + snv_referenceset_chm13_position['Seq'] + '_' + snv_referenceset_chm13_position['SNV_chm13_base'],
    snv_referenceset_chm13_position['Chromosome'] + '_' + snv_referenceset_chm13_position['End'].astype(str) + '_' + snv_referenceset_chm13_position['Seq'].apply(reverse_complement) + '_' + snv_referenceset_chm13_position['SNV_chm13_base'].apply(reverse_complement)
) # NOTE: it's a little mess for historical reason, but it works

snv_referenceset_chm13_position_set = set(snv_referenceset_chm13_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)
snv_referenceset_chm13_set = set(snv_referenceset_chm13_position["SNVid_chm13"].values)

# NOTE: DeepVariant/DeepSomatic only uses primary chromosomes. See https://github.com/google/deepvariant/blob/6fc7e0fc7edda9f84fdcb2e6d2ab965602a729c8/deepvariant/exclude_contigs.py#L4
primary_chrs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'] 

snv_referenceset_chm13_position_primary = snv_referenceset_chm13_position[snv_referenceset_chm13_position["Chromosome"].isin(primary_chrs)].reset_index(drop=True)
snv_referenceset_chm13_position_primary_set = set(snv_referenceset_chm13_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values) # NOTE: This was previously used for filtering (Deprecated and replaced by snv_referenceset_chm13_primary_set)

# INFO: Flagged SNV sites (DSA-based Flagged Set + overlapped SNVs-set surject-injected to CHM13)
snv_flagset_chm13_position = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV/SNV_Flagset_CHM13_peaks__surj_onto_CHM13_withTags_sorted.bed", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_flagset_chm13_position_set = set(snv_flagset_chm13_position[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values) # NOTE: 16,671

overlapped_leftover_snv_list_df = pd.read_table("/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/Benchmarking_Flagship/01.SNV/Overlapped_leftover_SNV_CHM13_peaks__surj_onto_CHM13_withTags_sorted.bed", sep="\t", names=["Chromosome", "Start", "End", "pgfbSNVid", "Length", "Strand"])
snv_flagset_chm13_position_set = snv_flagset_chm13_position_set.union(set(overlapped_leftover_snv_list_df[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values))

# NOTE: Subtracting Flagged SNV set from T2T-CHM13-surjected Reference (Primary) SNV set
condition1 = snv_referenceset_chm13_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).isin(snv_flagset_chm13_position_set)
condition2 = snv_referenceset_chm13_position_primary['SNVid'].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").isin(overlapped_leftover_snv_list_position_set) # NOTE: May seem redundant but necessary to ensure all overlapped SNVs are excluded from the DSA-based Reference Set

snv_referenceset_chm13_primary_set = set(snv_referenceset_chm13_position_primary[~condition1 & ~condition2]['SNVid_chm13'].values) # NOTE: 34,599

# ADDED: Non-satellite SNV set for T2T-CHM13
snv_referenceset_chm13_position_primary_pr = pr.from_dict({
    'Chromosome': snv_referenceset_chm13_position_primary["Chromosome"],
    'Start': snv_referenceset_chm13_position_primary["Start"],
    'End': snv_referenceset_chm13_position_primary["End"]
})

snv_referenceset_chm13_position_primary_non_satellite_pr = snv_referenceset_chm13_position_primary_pr.subtract(chm13_satellite_pr)
snv_referenceset_chm13_position_primary_non_satellite_position_set = set(snv_referenceset_chm13_position_primary_non_satellite_pr.df[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).values)

condition3 = snv_referenceset_chm13_position_primary[["Chromosome", "End"]].astype(str).apply('_'.join, axis=1).isin(snv_referenceset_chm13_position_primary_non_satellite_position_set)

snv_referenceset_chm13_primary_non_satellite_set = set(snv_referenceset_chm13_position_primary[~condition1 & ~condition2 & condition3]['SNVid_chm13'].values) # NOTE: 33,558

# ADDED: Non-satellite regions for T2T-CHM13 
chm13_fai = pd.read_table("/mmfs1/gscratch/stergachislab/assemblies/chm13v2.0.fa.fai", header=None, sep="\t", names=["Chromosome", "End", "_1", "_2", "_3"])
chm13_fai.insert(1, "Start", 0)
chm13_bed = chm13_fai[chm13_fai["Chromosome"].isin(primary_chrs)][["Chromosome", "Start", "End"]]

chm13_pr = pr.from_dict({
    'Chromosome': chm13_bed["Chromosome"],
    'Start': chm13_bed["Start"],
    'End': chm13_bed["End"]
})

chm13_non_satellite_pr = chm13_pr.subtract(chm13_satellite_pr)

#hg38_snv_set = set(snv_referenceset_hg38_position[snv_referenceset_hg38_position["SNVid_hg38"].isin(snv_referenceset_hg38_primary_set)]["SNVid"].values)
#chm13_snv_set = set(snv_referenceset_chm13_position[snv_referenceset_chm13_position["SNVid_chm13"].isin(snv_referenceset_chm13_primary_set)]["SNVid"].values)

# %%
# INFO: GRCh38
insilico_hg38_dir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/in_silico_mixture/hg38"

# INFO: With PON
bl_100x_hg38_pon_snv = read_vcf(f"{insilico_hg38_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_100x_hg38_pon_snv = bl_100x_hg38_pon_snv[bl_100x_hg38_pon_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)

bl_100x_hg38_pon_snv['SNVid_hg38'] = bl_100x_hg38_pon_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_hg38_pon_snv['POSid'] = bl_100x_hg38_pon_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_pon_100x = read_vcf(f"{insilico_hg38_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_pon_100x = read_vcf(f"{insilico_hg38_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_pon_100x = read_vcf(f"{insilico_hg38_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_hg38_pon_100x = blt_t1n4_hg38_pon_100x[blt_t1n4_hg38_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_pon_100x = blt_t1n9_hg38_pon_100x[blt_t1n9_hg38_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_pon_100x = blt_t1n19_hg38_pon_100x[blt_t1n19_hg38_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_hg38_pon_100x['SNVid_hg38'] = blt_t1n4_hg38_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_100x['SNVid_hg38'] = blt_t1n9_hg38_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_100x['SNVid_hg38'] = blt_t1n19_hg38_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_100x['POSid'] = blt_t1n4_hg38_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_100x['POSid'] = blt_t1n9_hg38_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_100x['POSid'] = blt_t1n19_hg38_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_pon_100x = blt_t1n4_hg38_pon_100x[~blt_t1n4_hg38_pon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_pon_100x = blt_t1n9_hg38_pon_100x[~blt_t1n9_hg38_pon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_pon_100x = blt_t1n19_hg38_pon_100x[~blt_t1n19_hg38_pon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n4_hg38_pon_10x = read_vcf(f"{insilico_hg38_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_hg38_pon_20x = read_vcf(f"{insilico_hg38_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_hg38_pon_40x = read_vcf(f"{insilico_hg38_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_hg38_pon_10x = blt_t1n4_hg38_pon_10x[blt_t1n4_hg38_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_hg38_pon_20x = blt_t1n4_hg38_pon_20x[blt_t1n4_hg38_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_hg38_pon_40x = blt_t1n4_hg38_pon_40x[blt_t1n4_hg38_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_hg38_pon_10x['SNVid_hg38'] = blt_t1n4_hg38_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_20x['SNVid_hg38'] = blt_t1n4_hg38_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_40x['SNVid_hg38'] = blt_t1n4_hg38_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_10x['POSid'] = blt_t1n4_hg38_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_20x['POSid'] = blt_t1n4_hg38_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_pon_40x['POSid'] = blt_t1n4_hg38_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_pon_10x = blt_t1n4_hg38_pon_10x[~blt_t1n4_hg38_pon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n4_hg38_pon_20x = blt_t1n4_hg38_pon_20x[~blt_t1n4_hg38_pon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n4_hg38_pon_40x = blt_t1n4_hg38_pon_40x[~blt_t1n4_hg38_pon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n9_hg38_pon_10x = read_vcf(f"{insilico_hg38_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_pon_20x = read_vcf(f"{insilico_hg38_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_pon_40x = read_vcf(f"{insilico_hg38_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n9_hg38_pon_10x = blt_t1n9_hg38_pon_10x[blt_t1n9_hg38_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_pon_20x = blt_t1n9_hg38_pon_20x[blt_t1n9_hg38_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_pon_40x = blt_t1n9_hg38_pon_40x[blt_t1n9_hg38_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n9_hg38_pon_10x['SNVid_hg38'] = blt_t1n9_hg38_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_20x['SNVid_hg38'] = blt_t1n9_hg38_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_40x['SNVid_hg38'] = blt_t1n9_hg38_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_10x['POSid'] = blt_t1n9_hg38_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_20x['POSid'] = blt_t1n9_hg38_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_pon_40x['POSid'] = blt_t1n9_hg38_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n9_hg38_pon_10x = blt_t1n9_hg38_pon_10x[~blt_t1n9_hg38_pon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_pon_20x = blt_t1n9_hg38_pon_20x[~blt_t1n9_hg38_pon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_pon_40x = blt_t1n9_hg38_pon_40x[~blt_t1n9_hg38_pon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n19_hg38_pon_10x = read_vcf(f"{insilico_hg38_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_pon_20x = read_vcf(f"{insilico_hg38_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_pon_40x = read_vcf(f"{insilico_hg38_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X_hg38_pon.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n19_hg38_pon_10x = blt_t1n19_hg38_pon_10x[blt_t1n19_hg38_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_pon_20x = blt_t1n19_hg38_pon_20x[blt_t1n19_hg38_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_pon_40x = blt_t1n19_hg38_pon_40x[blt_t1n19_hg38_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n19_hg38_pon_10x['SNVid_hg38'] = blt_t1n19_hg38_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_20x['SNVid_hg38'] = blt_t1n19_hg38_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_40x['SNVid_hg38'] = blt_t1n19_hg38_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_10x['POSid'] = blt_t1n19_hg38_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_20x['POSid'] = blt_t1n19_hg38_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_pon_40x['POSid'] = blt_t1n19_hg38_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n19_hg38_pon_10x = blt_t1n19_hg38_pon_10x[~blt_t1n19_hg38_pon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_pon_20x = blt_t1n19_hg38_pon_20x[~blt_t1n19_hg38_pon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_pon_40x = blt_t1n19_hg38_pon_40x[~blt_t1n19_hg38_pon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

# %%
# NOTE: Calculating Precision and Recall by SNVid_hg38 
#n_dsa_snv_nonsurjected_to_hg38 = len(snv_referenceset_position_set) - len(snv_referenceset_hg38_position_set)
n_dsa_snv_nonsurjected_to_hg38 = len(snv_referenceset) - len(snv_referenceset_hg38_primary_set)
# ADDED: 
n_dsa_snv_nonsurjected_to_hg38_satellite = len(snv_referenceset_non_satellite) - len(snv_referenceset_hg38_primary_non_satellite_set)

# INFO: T1N4 100X
#blt_t1n4_hg38_pon_100x_snv = set(blt_t1n4_hg38_pon_100x["POSid"].values) - set(bl_100x_hg38_pon_snv["POSid"].values)
blt_t1n4_hg38_pon_100x_snv = set(blt_t1n4_hg38_pon_100x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n4_hg38_pon_100x_snv_tp, blt_t1n4_hg38_pon_100x_snv_fp, blt_t1n4_hg38_pon_100x_snv_fn, blt_t1n4_hg38_pon_100x_snv_precision, blt_t1n4_hg38_pon_100x_snv_recall, blt_t1n4_hg38_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_pon_100x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 40X
blt_t1n4_hg38_pon_40x_snv = set(blt_t1n4_hg38_pon_40x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n4_hg38_pon_40x_snv_tp, blt_t1n4_hg38_pon_40x_snv_fp, blt_t1n4_hg38_pon_40x_snv_fn, blt_t1n4_hg38_pon_40x_snv_precision, blt_t1n4_hg38_pon_40x_snv_recall, blt_t1n4_hg38_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_pon_40x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 20X
blt_t1n4_hg38_pon_20x_snv = set(blt_t1n4_hg38_pon_20x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n4_hg38_pon_20x_snv_tp, blt_t1n4_hg38_pon_20x_snv_fp, blt_t1n4_hg38_pon_20x_snv_fn, blt_t1n4_hg38_pon_20x_snv_precision, blt_t1n4_hg38_pon_20x_snv_recall, blt_t1n4_hg38_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_pon_20x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 10X
blt_t1n4_hg38_pon_10x_snv = set(blt_t1n4_hg38_pon_10x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n4_hg38_pon_10x_snv_tp, blt_t1n4_hg38_pon_10x_snv_fp, blt_t1n4_hg38_pon_10x_snv_fn, blt_t1n4_hg38_pon_10x_snv_precision, blt_t1n4_hg38_pon_10x_snv_recall, blt_t1n4_hg38_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_pon_10x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 100X
blt_t1n9_hg38_pon_100x_snv = set(blt_t1n9_hg38_pon_100x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n9_hg38_pon_100x_snv_tp, blt_t1n9_hg38_pon_100x_snv_fp, blt_t1n9_hg38_pon_100x_snv_fn, blt_t1n9_hg38_pon_100x_snv_precision, blt_t1n9_hg38_pon_100x_snv_recall, blt_t1n9_hg38_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_pon_100x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 40X
blt_t1n9_hg38_pon_40x_snv = set(blt_t1n9_hg38_pon_40x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n9_hg38_pon_40x_snv_tp, blt_t1n9_hg38_pon_40x_snv_fp, blt_t1n9_hg38_pon_40x_snv_fn, blt_t1n9_hg38_pon_40x_snv_precision, blt_t1n9_hg38_pon_40x_snv_recall, blt_t1n9_hg38_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_pon_40x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 20X
blt_t1n9_hg38_pon_20x_snv = set(blt_t1n9_hg38_pon_20x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n9_hg38_pon_20x_snv_tp, blt_t1n9_hg38_pon_20x_snv_fp, blt_t1n9_hg38_pon_20x_snv_fn, blt_t1n9_hg38_pon_20x_snv_precision, blt_t1n9_hg38_pon_20x_snv_recall, blt_t1n9_hg38_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_pon_20x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 10X
blt_t1n9_hg38_pon_10x_snv = set(blt_t1n9_hg38_pon_10x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n9_hg38_pon_10x_snv_tp, blt_t1n9_hg38_pon_10x_snv_fp, blt_t1n9_hg38_pon_10x_snv_fn, blt_t1n9_hg38_pon_10x_snv_precision, blt_t1n9_hg38_pon_10x_snv_recall, blt_t1n9_hg38_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_pon_10x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 100X
blt_t1n19_hg38_pon_100x_snv = set(blt_t1n19_hg38_pon_100x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n19_hg38_pon_100x_snv_tp, blt_t1n19_hg38_pon_100x_snv_fp, blt_t1n19_hg38_pon_100x_snv_fn, blt_t1n19_hg38_pon_100x_snv_precision, blt_t1n19_hg38_pon_100x_snv_recall, blt_t1n19_hg38_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_pon_100x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 40X
blt_t1n19_hg38_pon_40x_snv = set(blt_t1n19_hg38_pon_40x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n19_hg38_pon_40x_snv_tp, blt_t1n19_hg38_pon_40x_snv_fp, blt_t1n19_hg38_pon_40x_snv_fn, blt_t1n19_hg38_pon_40x_snv_precision, blt_t1n19_hg38_pon_40x_snv_recall, blt_t1n19_hg38_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_pon_40x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 20X
blt_t1n19_hg38_pon_20x_snv = set(blt_t1n19_hg38_pon_20x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n19_hg38_pon_20x_snv_tp, blt_t1n19_hg38_pon_20x_snv_fp, blt_t1n19_hg38_pon_20x_snv_fn, blt_t1n19_hg38_pon_20x_snv_precision, blt_t1n19_hg38_pon_20x_snv_recall, blt_t1n19_hg38_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_pon_20x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 10X
blt_t1n19_hg38_pon_10x_snv = set(blt_t1n19_hg38_pon_10x["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv["SNVid_hg38"].values)
blt_t1n19_hg38_pon_10x_snv_tp, blt_t1n19_hg38_pon_10x_snv_fp, blt_t1n19_hg38_pon_10x_snv_fn, blt_t1n19_hg38_pon_10x_snv_precision, blt_t1n19_hg38_pon_10x_snv_recall, blt_t1n19_hg38_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_pon_10x_snv, snv_referenceset_hg38_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)

# %%
# INFO: Without PON filtering
bl_100x_hg38_nonpon_snv = read_vcf(f"{insilico_hg38_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_100x_hg38_nonpon_snv = bl_100x_hg38_nonpon_snv[bl_100x_hg38_nonpon_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)

bl_100x_hg38_nonpon_snv['SNVid'] = bl_100x_hg38_nonpon_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_hg38_nonpon_snv['POSid'] = bl_100x_hg38_nonpon_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_nonpon_100x = read_vcf(f"{insilico_hg38_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_nonpon_100x = read_vcf(f"{insilico_hg38_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_nonpon_100x = read_vcf(f"{insilico_hg38_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_hg38_nonpon_100x = blt_t1n4_hg38_nonpon_100x[blt_t1n4_hg38_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_100x = blt_t1n9_hg38_nonpon_100x[blt_t1n9_hg38_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_100x = blt_t1n19_hg38_nonpon_100x[blt_t1n19_hg38_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_hg38_nonpon_100x['SNVid'] = blt_t1n4_hg38_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_100x['SNVid'] = blt_t1n9_hg38_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_100x['SNVid'] = blt_t1n19_hg38_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_100x['POSid'] = blt_t1n4_hg38_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_100x['POSid'] = blt_t1n9_hg38_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_100x['POSid'] = blt_t1n19_hg38_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_nonpon_100x = blt_t1n4_hg38_nonpon_100x[~blt_t1n4_hg38_nonpon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_100x = blt_t1n9_hg38_nonpon_100x[~blt_t1n9_hg38_nonpon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_100x = blt_t1n19_hg38_nonpon_100x[~blt_t1n19_hg38_nonpon_100x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n4_hg38_nonpon_10x = read_vcf(f"{insilico_hg38_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_hg38_nonpon_20x = read_vcf(f"{insilico_hg38_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_hg38_nonpon_40x = read_vcf(f"{insilico_hg38_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_hg38_nonpon_10x = blt_t1n4_hg38_nonpon_10x[blt_t1n4_hg38_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_hg38_nonpon_20x = blt_t1n4_hg38_nonpon_20x[blt_t1n4_hg38_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_hg38_nonpon_40x = blt_t1n4_hg38_nonpon_40x[blt_t1n4_hg38_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_hg38_nonpon_10x['SNVid'] = blt_t1n4_hg38_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_20x['SNVid'] = blt_t1n4_hg38_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_40x['SNVid'] = blt_t1n4_hg38_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_10x['POSid'] = blt_t1n4_hg38_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_20x['POSid'] = blt_t1n4_hg38_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_hg38_nonpon_40x['POSid'] = blt_t1n4_hg38_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_nonpon_10x = blt_t1n4_hg38_nonpon_10x[~blt_t1n4_hg38_nonpon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n4_hg38_nonpon_20x = blt_t1n4_hg38_nonpon_20x[~blt_t1n4_hg38_nonpon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n4_hg38_nonpon_40x = blt_t1n4_hg38_nonpon_40x[~blt_t1n4_hg38_nonpon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n9_hg38_nonpon_10x = read_vcf(f"{insilico_hg38_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_nonpon_20x = read_vcf(f"{insilico_hg38_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_nonpon_40x = read_vcf(f"{insilico_hg38_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n9_hg38_nonpon_10x = blt_t1n9_hg38_nonpon_10x[blt_t1n9_hg38_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_20x = blt_t1n9_hg38_nonpon_20x[blt_t1n9_hg38_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_40x = blt_t1n9_hg38_nonpon_40x[blt_t1n9_hg38_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n9_hg38_nonpon_10x['SNVid'] = blt_t1n9_hg38_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_20x['SNVid'] = blt_t1n9_hg38_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_40x['SNVid'] = blt_t1n9_hg38_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_10x['POSid'] = blt_t1n9_hg38_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_20x['POSid'] = blt_t1n9_hg38_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_nonpon_40x['POSid'] = blt_t1n9_hg38_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n9_hg38_nonpon_10x = blt_t1n9_hg38_nonpon_10x[~blt_t1n9_hg38_nonpon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_20x = blt_t1n9_hg38_nonpon_20x[~blt_t1n9_hg38_nonpon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_nonpon_40x = blt_t1n9_hg38_nonpon_40x[~blt_t1n9_hg38_nonpon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

blt_t1n19_hg38_nonpon_10x = read_vcf(f"{insilico_hg38_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_nonpon_20x = read_vcf(f"{insilico_hg38_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_nonpon_40x = read_vcf(f"{insilico_hg38_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n19_hg38_nonpon_10x = blt_t1n19_hg38_nonpon_10x[blt_t1n19_hg38_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_20x = blt_t1n19_hg38_nonpon_20x[blt_t1n19_hg38_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_40x = blt_t1n19_hg38_nonpon_40x[blt_t1n19_hg38_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n19_hg38_nonpon_10x['SNVid'] = blt_t1n19_hg38_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_20x['SNVid'] = blt_t1n19_hg38_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_40x['SNVid'] = blt_t1n19_hg38_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_10x['POSid'] = blt_t1n19_hg38_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_20x['POSid'] = blt_t1n19_hg38_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_nonpon_40x['POSid'] = blt_t1n19_hg38_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n19_hg38_nonpon_10x = blt_t1n19_hg38_nonpon_10x[~blt_t1n19_hg38_nonpon_10x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_20x = blt_t1n19_hg38_nonpon_20x[~blt_t1n19_hg38_nonpon_20x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_nonpon_40x = blt_t1n19_hg38_nonpon_40x[~blt_t1n19_hg38_nonpon_40x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

# INFO: T1N4 100X
# NOTE: position-resolution (POSid) BY DESIGN here — hg38 non-PON is scored at chrom+pos,
#       unlike hg38-PON / chm13 which use allele resolution (SNVid). Confirmed intentional 2026-07-23.
blt_t1n4_hg38_nonpon_100x_snv = set(blt_t1n4_hg38_nonpon_100x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n4_hg38_nonpon_100x_snv_tp, blt_t1n4_hg38_nonpon_100x_snv_fp, blt_t1n4_hg38_nonpon_100x_snv_fn, blt_t1n4_hg38_nonpon_100x_snv_precision, blt_t1n4_hg38_nonpon_100x_snv_recall, blt_t1n4_hg38_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_nonpon_100x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 40X
blt_t1n4_hg38_nonpon_40x_snv = set(blt_t1n4_hg38_nonpon_40x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n4_hg38_nonpon_40x_snv_tp, blt_t1n4_hg38_nonpon_40x_snv_fp, blt_t1n4_hg38_nonpon_40x_snv_fn, blt_t1n4_hg38_nonpon_40x_snv_precision, blt_t1n4_hg38_nonpon_40x_snv_recall, blt_t1n4_hg38_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_nonpon_40x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 20X
blt_t1n4_hg38_nonpon_20x_snv = set(blt_t1n4_hg38_nonpon_20x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n4_hg38_nonpon_20x_snv_tp, blt_t1n4_hg38_nonpon_20x_snv_fp, blt_t1n4_hg38_nonpon_20x_snv_fn, blt_t1n4_hg38_nonpon_20x_snv_precision, blt_t1n4_hg38_nonpon_20x_snv_recall, blt_t1n4_hg38_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_nonpon_20x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N4 10X
blt_t1n4_hg38_nonpon_10x_snv = set(blt_t1n4_hg38_nonpon_10x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n4_hg38_nonpon_10x_snv_tp, blt_t1n4_hg38_nonpon_10x_snv_fp, blt_t1n4_hg38_nonpon_10x_snv_fn, blt_t1n4_hg38_nonpon_10x_snv_precision, blt_t1n4_hg38_nonpon_10x_snv_recall, blt_t1n4_hg38_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_nonpon_10x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 100X
blt_t1n9_hg38_nonpon_100x_snv = set(blt_t1n9_hg38_nonpon_100x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n9_hg38_nonpon_100x_snv_tp, blt_t1n9_hg38_nonpon_100x_snv_fp, blt_t1n9_hg38_nonpon_100x_snv_fn, blt_t1n9_hg38_nonpon_100x_snv_precision, blt_t1n9_hg38_nonpon_100x_snv_recall, blt_t1n9_hg38_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_nonpon_100x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 40X
blt_t1n9_hg38_nonpon_40x_snv = set(blt_t1n9_hg38_nonpon_40x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n9_hg38_nonpon_40x_snv_tp, blt_t1n9_hg38_nonpon_40x_snv_fp, blt_t1n9_hg38_nonpon_40x_snv_fn, blt_t1n9_hg38_nonpon_40x_snv_precision, blt_t1n9_hg38_nonpon_40x_snv_recall, blt_t1n9_hg38_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_nonpon_40x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 20X
blt_t1n9_hg38_nonpon_20x_snv = set(blt_t1n9_hg38_nonpon_20x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n9_hg38_nonpon_20x_snv_tp, blt_t1n9_hg38_nonpon_20x_snv_fp, blt_t1n9_hg38_nonpon_20x_snv_fn, blt_t1n9_hg38_nonpon_20x_snv_precision, blt_t1n9_hg38_nonpon_20x_snv_recall, blt_t1n9_hg38_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_nonpon_20x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 10X
blt_t1n9_hg38_nonpon_10x_snv = set(blt_t1n9_hg38_nonpon_10x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n9_hg38_nonpon_10x_snv_tp, blt_t1n9_hg38_nonpon_10x_snv_fp, blt_t1n9_hg38_nonpon_10x_snv_fn, blt_t1n9_hg38_nonpon_10x_snv_precision, blt_t1n9_hg38_nonpon_10x_snv_recall, blt_t1n9_hg38_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_nonpon_10x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 100X
blt_t1n19_hg38_nonpon_100x_snv = set(blt_t1n19_hg38_nonpon_100x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n19_hg38_nonpon_100x_snv_tp, blt_t1n19_hg38_nonpon_100x_snv_fp, blt_t1n19_hg38_nonpon_100x_snv_fn, blt_t1n19_hg38_nonpon_100x_snv_precision, blt_t1n19_hg38_nonpon_100x_snv_recall, blt_t1n19_hg38_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_nonpon_100x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 40X
blt_t1n19_hg38_nonpon_40x_snv = set(blt_t1n19_hg38_nonpon_40x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n19_hg38_nonpon_40x_snv_tp, blt_t1n19_hg38_nonpon_40x_snv_fp, blt_t1n19_hg38_nonpon_40x_snv_fn, blt_t1n19_hg38_nonpon_40x_snv_precision, blt_t1n19_hg38_nonpon_40x_snv_recall, blt_t1n19_hg38_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_nonpon_40x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 20X
blt_t1n19_hg38_nonpon_20x_snv = set(blt_t1n19_hg38_nonpon_20x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n19_hg38_nonpon_20x_snv_tp, blt_t1n19_hg38_nonpon_20x_snv_fp, blt_t1n19_hg38_nonpon_20x_snv_fn, blt_t1n19_hg38_nonpon_20x_snv_precision, blt_t1n19_hg38_nonpon_20x_snv_recall, blt_t1n19_hg38_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_nonpon_20x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 10X
blt_t1n19_hg38_nonpon_10x_snv = set(blt_t1n19_hg38_nonpon_10x["POSid"].values) - set(bl_100x_hg38_nonpon_snv["POSid"].values)
blt_t1n19_hg38_nonpon_10x_snv_tp, blt_t1n19_hg38_nonpon_10x_snv_fp, blt_t1n19_hg38_nonpon_10x_snv_fn, blt_t1n19_hg38_nonpon_10x_snv_precision, blt_t1n19_hg38_nonpon_10x_snv_recall, blt_t1n19_hg38_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_nonpon_10x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# %%
# ADDED: Illumina short-read (Diploid 100X, haploid 50X) for evaluating mSNV discovery performance - GRCh38
# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures for GRCh38
platform="Illumina"
bl_sr_hg38_50x_snv = read_vcf(f"{insilico_hg38_dir}/N_ONLY/{platform}/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_sr_50X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

bl_sr_hg38_50x_snv = bl_sr_hg38_50x_snv[bl_sr_hg38_50x_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)
bl_sr_hg38_50x_snv['SNVid'] = bl_sr_hg38_50x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_sr_hg38_50x_snv['POSid'] = bl_sr_hg38_50x_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_sr_50x = read_vcf(f"{insilico_hg38_dir}/T1N4/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_sr_50X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_hg38_sr_50x = read_vcf(f"{insilico_hg38_dir}/T1N9/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_sr_50X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_hg38_sr_50x = read_vcf(f"{insilico_hg38_dir}/T1N19/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_sr_50X_hg38.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_hg38_sr_50x = blt_t1n4_hg38_sr_50x[blt_t1n4_hg38_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_hg38_sr_50x = blt_t1n9_hg38_sr_50x[blt_t1n9_hg38_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_hg38_sr_50x = blt_t1n19_hg38_sr_50x[blt_t1n19_hg38_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_hg38_sr_50x['SNVid'] = blt_t1n4_hg38_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_sr_50x['SNVid'] = blt_t1n9_hg38_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_sr_50x['SNVid'] = blt_t1n19_hg38_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_sr_50x["POSid"] = blt_t1n4_hg38_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_hg38_sr_50x["POSid"] = blt_t1n9_hg38_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_hg38_sr_50x["POSid"] = blt_t1n19_hg38_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_hg38_sr_50x = blt_t1n4_hg38_sr_50x[~blt_t1n4_hg38_sr_50x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n9_hg38_sr_50x = blt_t1n9_hg38_sr_50x[~blt_t1n9_hg38_sr_50x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)
blt_t1n19_hg38_sr_50x = blt_t1n19_hg38_sr_50x[~blt_t1n19_hg38_sr_50x["POSid"].isin(snv_flagset_hg38_position_set)].reset_index(drop=True)

# INFO: Calculating Precision and Recall in the DSA space but FOR SHORT-READ (Illumina) sequencing data
# INFO: T1N4 50X
# NOTE: position-resolution (POSid) BY DESIGN here — hg38 short-read is scored at chrom+pos,
#       unlike hg38-PON / chm13 which use allele resolution (SNVid). Confirmed intentional 2026-07-23.
blt_t1n4_hg38_sr_50x_snv = set(blt_t1n4_hg38_sr_50x["POSid"].values) - set(bl_sr_hg38_50x_snv["POSid"].values)
blt_t1n4_hg38_sr_50x_snv_tp, blt_t1n4_hg38_sr_50x_snv_fp, blt_t1n4_hg38_sr_50x_snv_fn, blt_t1n4_hg38_sr_50x_snv_precision, blt_t1n4_hg38_sr_50x_snv_recall, blt_t1n4_hg38_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_sr_50x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N9 50X
blt_t1n9_hg38_sr_50x_snv = set(blt_t1n9_hg38_sr_50x["POSid"].values) - set(bl_sr_hg38_50x_snv["POSid"].values)
blt_t1n9_hg38_sr_50x_snv_tp, blt_t1n9_hg38_sr_50x_snv_fp, blt_t1n9_hg38_sr_50x_snv_fn, blt_t1n9_hg38_sr_50x_snv_precision, blt_t1n9_hg38_sr_50x_snv_recall, blt_t1n9_hg38_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_sr_50x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# INFO: T1N19 50X
blt_t1n19_hg38_sr_50x_snv = set(blt_t1n19_hg38_sr_50x["POSid"].values) - set(bl_sr_hg38_50x_snv["POSid"].values)
blt_t1n19_hg38_sr_50x_snv_tp, blt_t1n19_hg38_sr_50x_snv_fp, blt_t1n19_hg38_sr_50x_snv_fn, blt_t1n19_hg38_sr_50x_snv_precision, blt_t1n19_hg38_sr_50x_snv_recall, blt_t1n19_hg38_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_sr_50x_snv, snv_referenceset_hg38_position_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38)


# %%
# ADDED: Non-Satellite Regions of the GRCh38
# INFO: PacBio
bl_100x_hg38_pon_snv_non_satellite = vcf_in_pyranges_interval(bl_100x_hg38_pon_snv, hg38_non_satellite_pr, id="SNVid_hg38")

blt_t1n4_hg38_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_hg38_pon_20x, hg38_non_satellite_pr, id="SNVid_hg38")
blt_t1n9_hg38_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_hg38_pon_20x, hg38_non_satellite_pr, id="SNVid_hg38")
blt_t1n19_hg38_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_hg38_pon_20x, hg38_non_satellite_pr, id="SNVid_hg38")

# INFO: T1N4 20X
blt_t1n4_hg38_pon_20x_non_satellite_snv = set(blt_t1n4_hg38_pon_20x_non_satellite["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv_non_satellite["SNVid_hg38"].values)
blt_t1n4_hg38_pon_20x_non_satellite_snv_tp, blt_t1n4_hg38_pon_20x_non_satellite_snv_fp, blt_t1n4_hg38_pon_20x_non_satellite_snv_fn, blt_t1n4_hg38_pon_20x_non_satellite_snv_precision, blt_t1n4_hg38_pon_20x_non_satellite_snv_recall, blt_t1n4_hg38_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_pon_20x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)

# INFO: T1N9 20X
blt_t1n9_hg38_pon_20x_non_satellite_snv = set(blt_t1n9_hg38_pon_20x_non_satellite["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv_non_satellite["SNVid_hg38"].values)
blt_t1n9_hg38_pon_20x_non_satellite_snv_tp, blt_t1n9_hg38_pon_20x_non_satellite_snv_fp, blt_t1n9_hg38_pon_20x_non_satellite_snv_fn, blt_t1n9_hg38_pon_20x_non_satellite_snv_precision, blt_t1n9_hg38_pon_20x_non_satellite_snv_recall, blt_t1n9_hg38_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_pon_20x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)

# INFO: T1N19 20X
blt_t1n19_hg38_pon_20x_non_satellite_snv = set(blt_t1n19_hg38_pon_20x_non_satellite["SNVid_hg38"].values) - set(bl_100x_hg38_pon_snv_non_satellite["SNVid_hg38"].values)
blt_t1n19_hg38_pon_20x_non_satellite_snv_tp, blt_t1n19_hg38_pon_20x_non_satellite_snv_fp, blt_t1n19_hg38_pon_20x_non_satellite_snv_fn, blt_t1n19_hg38_pon_20x_non_satellite_snv_precision, blt_t1n19_hg38_pon_20x_non_satellite_snv_recall, blt_t1n19_hg38_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_pon_20x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)


# INFO: Illumina Short-read
bl_sr_hg38_50x_snv_non_satellite = vcf_in_pyranges_interval(bl_sr_hg38_50x_snv, hg38_non_satellite_pr, id="POSid")

blt_t1n4_hg38_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_hg38_sr_50x, hg38_non_satellite_pr, id="POSid")
blt_t1n9_hg38_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_hg38_sr_50x, hg38_non_satellite_pr, id="POSid")
blt_t1n19_hg38_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_hg38_sr_50x, hg38_non_satellite_pr, id="POSid")

# INFO: T1N4 50X
blt_t1n4_hg38_sr_50x_non_satellite_snv = set(blt_t1n4_hg38_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_hg38_50x_snv_non_satellite["SNVid"].values)
blt_t1n4_hg38_sr_50x_non_satellite_snv_tp, blt_t1n4_hg38_sr_50x_non_satellite_snv_fp, blt_t1n4_hg38_sr_50x_non_satellite_snv_fn, blt_t1n4_hg38_sr_50x_non_satellite_snv_precision, blt_t1n4_hg38_sr_50x_non_satellite_snv_recall, blt_t1n4_hg38_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n4_hg38_sr_50x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)

# INFO: T1N9 50X
blt_t1n9_hg38_sr_50x_non_satellite_snv = set(blt_t1n9_hg38_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_hg38_50x_snv_non_satellite["SNVid"].values)
blt_t1n9_hg38_sr_50x_non_satellite_snv_tp, blt_t1n9_hg38_sr_50x_non_satellite_snv_fp, blt_t1n9_hg38_sr_50x_non_satellite_snv_fn, blt_t1n9_hg38_sr_50x_non_satellite_snv_precision, blt_t1n9_hg38_sr_50x_non_satellite_snv_recall, blt_t1n9_hg38_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n9_hg38_sr_50x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)

# INFO: T1N19 50X
blt_t1n19_hg38_sr_50x_non_satellite_snv = set(blt_t1n19_hg38_sr_50x_non_satellite["SNVid"].values) - set(bl_sr_hg38_50x_snv_non_satellite["SNVid"].values)
blt_t1n19_hg38_sr_50x_non_satellite_snv_tp, blt_t1n19_hg38_sr_50x_non_satellite_snv_fp, blt_t1n19_hg38_sr_50x_non_satellite_snv_fn, blt_t1n19_hg38_sr_50x_non_satellite_snv_precision, blt_t1n19_hg38_sr_50x_non_satellite_snv_recall, blt_t1n19_hg38_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n19_hg38_sr_50x_non_satellite_snv, snv_referenceset_hg38_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_hg38_satellite)

# %%

############################################################################################################################
############################################################################################################################
############################################################################################################################
# INFO: Evaluating SNV detection performance using T2T-CHM13 genome as well for the Revision ###############################
############################################################################################################################
############################################################################################################################
############################################################################################################################

# %% [markdown]
# INFO:
# DSA-based Reference SNV-set injected to DSG and surjected to T2T-CHM13

# %%
insilico_chm13_dir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/in_silico_mixture/chm13"
# INFO: With PON
bl_100x_chm13_pon_snv = read_vcf(f"{insilico_chm13_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

bl_100x_chm13_pon_snv = bl_100x_chm13_pon_snv[bl_100x_chm13_pon_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)

bl_100x_chm13_pon_snv['SNVid_chm13'] = bl_100x_chm13_pon_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_chm13_pon_snv['POSid'] = bl_100x_chm13_pon_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_pon_100x = read_vcf(f"{insilico_chm13_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_pon_100x = read_vcf(f"{insilico_chm13_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_pon_100x = read_vcf(f"{insilico_chm13_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n4_chm13_pon_100x = blt_t1n4_chm13_pon_100x[blt_t1n4_chm13_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_pon_100x = blt_t1n9_chm13_pon_100x[blt_t1n9_chm13_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_pon_100x = blt_t1n19_chm13_pon_100x[blt_t1n19_chm13_pon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_chm13_pon_100x['SNVid_chm13'] = blt_t1n4_chm13_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_100x['SNVid_chm13'] = blt_t1n9_chm13_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_100x['SNVid_chm13'] = blt_t1n19_chm13_pon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_100x['POSid'] = blt_t1n4_chm13_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_100x['POSid'] = blt_t1n9_chm13_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_100x['POSid'] = blt_t1n19_chm13_pon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_pon_100x = blt_t1n4_chm13_pon_100x[~blt_t1n4_chm13_pon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_pon_100x = blt_t1n9_chm13_pon_100x[~blt_t1n9_chm13_pon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_pon_100x = blt_t1n19_chm13_pon_100x[~blt_t1n19_chm13_pon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n4_chm13_pon_10x = read_vcf(f"{insilico_chm13_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n4_chm13_pon_20x = read_vcf(f"{insilico_chm13_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n4_chm13_pon_40x = read_vcf(f"{insilico_chm13_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n4_chm13_pon_10x = blt_t1n4_chm13_pon_10x[blt_t1n4_chm13_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_chm13_pon_20x = blt_t1n4_chm13_pon_20x[blt_t1n4_chm13_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_chm13_pon_40x = blt_t1n4_chm13_pon_40x[blt_t1n4_chm13_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_chm13_pon_10x['SNVid_chm13'] = blt_t1n4_chm13_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_20x['SNVid_chm13'] = blt_t1n4_chm13_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_40x['SNVid_chm13'] = blt_t1n4_chm13_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_10x['POSid'] = blt_t1n4_chm13_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_20x['POSid'] = blt_t1n4_chm13_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_pon_40x['POSid'] = blt_t1n4_chm13_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_pon_10x = blt_t1n4_chm13_pon_10x[~blt_t1n4_chm13_pon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n4_chm13_pon_20x = blt_t1n4_chm13_pon_20x[~blt_t1n4_chm13_pon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n4_chm13_pon_40x = blt_t1n4_chm13_pon_40x[~blt_t1n4_chm13_pon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n9_chm13_pon_10x = read_vcf(f"{insilico_chm13_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_pon_20x = read_vcf(f"{insilico_chm13_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_pon_40x = read_vcf(f"{insilico_chm13_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n9_chm13_pon_10x = blt_t1n9_chm13_pon_10x[blt_t1n9_chm13_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_pon_20x = blt_t1n9_chm13_pon_20x[blt_t1n9_chm13_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_pon_40x = blt_t1n9_chm13_pon_40x[blt_t1n9_chm13_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n9_chm13_pon_10x['SNVid_chm13'] = blt_t1n9_chm13_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_20x['SNVid_chm13'] = blt_t1n9_chm13_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_40x['SNVid_chm13'] = blt_t1n9_chm13_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_10x['POSid'] = blt_t1n9_chm13_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_20x['POSid'] = blt_t1n9_chm13_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_pon_40x['POSid'] = blt_t1n9_chm13_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n9_chm13_pon_10x = blt_t1n9_chm13_pon_10x[~blt_t1n9_chm13_pon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_pon_20x = blt_t1n9_chm13_pon_20x[~blt_t1n9_chm13_pon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_pon_40x = blt_t1n9_chm13_pon_40x[~blt_t1n9_chm13_pon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n19_chm13_pon_10x = read_vcf(f"{insilico_chm13_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_pon_20x = read_vcf(f"{insilico_chm13_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_pon_40x = read_vcf(f"{insilico_chm13_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n19_chm13_pon_10x = blt_t1n19_chm13_pon_10x[blt_t1n19_chm13_pon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_pon_20x = blt_t1n19_chm13_pon_20x[blt_t1n19_chm13_pon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_pon_40x = blt_t1n19_chm13_pon_40x[blt_t1n19_chm13_pon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n19_chm13_pon_10x['SNVid_chm13'] = blt_t1n19_chm13_pon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_20x['SNVid_chm13'] = blt_t1n19_chm13_pon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_40x['SNVid_chm13'] = blt_t1n19_chm13_pon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_10x['POSid'] = blt_t1n19_chm13_pon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_20x['POSid'] = blt_t1n19_chm13_pon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_pon_40x['POSid'] = blt_t1n19_chm13_pon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n19_chm13_pon_10x = blt_t1n19_chm13_pon_10x[~blt_t1n19_chm13_pon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_pon_20x = blt_t1n19_chm13_pon_20x[~blt_t1n19_chm13_pon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_pon_40x = blt_t1n19_chm13_pon_40x[~blt_t1n19_chm13_pon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

# %%
# NOTE: Calculating Precision and Recall by SNVid_chm13 
n_dsa_snv_nonsurjected_to_chm13 = len(snv_referenceset) - len(snv_referenceset_chm13_primary_set)
# ADDED: 
n_dsa_snv_nonsurjected_to_chm13_satellite = len(snv_referenceset_non_satellite) - len(snv_referenceset_chm13_primary_non_satellite_set)


# INFO: T1N4 100X
blt_t1n4_chm13_pon_100x_snv = set(blt_t1n4_chm13_pon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_pon_100x_snv_tp, blt_t1n4_chm13_pon_100x_snv_fp, blt_t1n4_chm13_pon_100x_snv_fn, blt_t1n4_chm13_pon_100x_snv_precision, blt_t1n4_chm13_pon_100x_snv_recall, blt_t1n4_chm13_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_pon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 40X
blt_t1n4_chm13_pon_40x_snv = set(blt_t1n4_chm13_pon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_pon_40x_snv_tp, blt_t1n4_chm13_pon_40x_snv_fp, blt_t1n4_chm13_pon_40x_snv_fn, blt_t1n4_chm13_pon_40x_snv_precision, blt_t1n4_chm13_pon_40x_snv_recall, blt_t1n4_chm13_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_pon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 20X
blt_t1n4_chm13_pon_20x_snv = set(blt_t1n4_chm13_pon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_pon_20x_snv_tp, blt_t1n4_chm13_pon_20x_snv_fp, blt_t1n4_chm13_pon_20x_snv_fn, blt_t1n4_chm13_pon_20x_snv_precision, blt_t1n4_chm13_pon_20x_snv_recall, blt_t1n4_chm13_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_pon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 10X
blt_t1n4_chm13_pon_10x_snv = set(blt_t1n4_chm13_pon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_pon_10x_snv_tp, blt_t1n4_chm13_pon_10x_snv_fp, blt_t1n4_chm13_pon_10x_snv_fn, blt_t1n4_chm13_pon_10x_snv_precision, blt_t1n4_chm13_pon_10x_snv_recall, blt_t1n4_chm13_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_pon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 100X
blt_t1n9_chm13_pon_100x_snv = set(blt_t1n9_chm13_pon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_pon_100x_snv_tp, blt_t1n9_chm13_pon_100x_snv_fp, blt_t1n9_chm13_pon_100x_snv_fn, blt_t1n9_chm13_pon_100x_snv_precision, blt_t1n9_chm13_pon_100x_snv_recall, blt_t1n9_chm13_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_pon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 40X
blt_t1n9_chm13_pon_40x_snv = set(blt_t1n9_chm13_pon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_pon_40x_snv_tp, blt_t1n9_chm13_pon_40x_snv_fp, blt_t1n9_chm13_pon_40x_snv_fn, blt_t1n9_chm13_pon_40x_snv_precision, blt_t1n9_chm13_pon_40x_snv_recall, blt_t1n9_chm13_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_pon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 20X
blt_t1n9_chm13_pon_20x_snv = set(blt_t1n9_chm13_pon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_pon_20x_snv_tp, blt_t1n9_chm13_pon_20x_snv_fp, blt_t1n9_chm13_pon_20x_snv_fn, blt_t1n9_chm13_pon_20x_snv_precision, blt_t1n9_chm13_pon_20x_snv_recall, blt_t1n9_chm13_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_pon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 10X
blt_t1n9_chm13_pon_10x_snv = set(blt_t1n9_chm13_pon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_pon_10x_snv_tp, blt_t1n9_chm13_pon_10x_snv_fp, blt_t1n9_chm13_pon_10x_snv_fn, blt_t1n9_chm13_pon_10x_snv_precision, blt_t1n9_chm13_pon_10x_snv_recall, blt_t1n9_chm13_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_pon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 100X
blt_t1n19_chm13_pon_100x_snv = set(blt_t1n19_chm13_pon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_pon_100x_snv_tp, blt_t1n19_chm13_pon_100x_snv_fp, blt_t1n19_chm13_pon_100x_snv_fn, blt_t1n19_chm13_pon_100x_snv_precision, blt_t1n19_chm13_pon_100x_snv_recall, blt_t1n19_chm13_pon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_pon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 40X
blt_t1n19_chm13_pon_40x_snv = set(blt_t1n19_chm13_pon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_pon_40x_snv_tp, blt_t1n19_chm13_pon_40x_snv_fp, blt_t1n19_chm13_pon_40x_snv_fn, blt_t1n19_chm13_pon_40x_snv_precision, blt_t1n19_chm13_pon_40x_snv_recall, blt_t1n19_chm13_pon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_pon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 20X
blt_t1n19_chm13_pon_20x_snv = set(blt_t1n19_chm13_pon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_pon_20x_snv_tp, blt_t1n19_chm13_pon_20x_snv_fp, blt_t1n19_chm13_pon_20x_snv_fn, blt_t1n19_chm13_pon_20x_snv_precision, blt_t1n19_chm13_pon_20x_snv_recall, blt_t1n19_chm13_pon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_pon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 10X
blt_t1n19_chm13_pon_10x_snv = set(blt_t1n19_chm13_pon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_pon_10x_snv_tp, blt_t1n19_chm13_pon_10x_snv_fp, blt_t1n19_chm13_pon_10x_snv_fn, blt_t1n19_chm13_pon_10x_snv_precision, blt_t1n19_chm13_pon_10x_snv_recall, blt_t1n19_chm13_pon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_pon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)



# %%
# INFO: Without PON filtering
bl_100x_chm13_nonpon_snv = read_vcf(f"{insilico_chm13_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
bl_100x_chm13_nonpon_snv = bl_100x_chm13_nonpon_snv[bl_100x_chm13_nonpon_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)

bl_100x_chm13_nonpon_snv['SNVid_chm13'] = bl_100x_chm13_nonpon_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_chm13_nonpon_snv['POSid'] = bl_100x_chm13_nonpon_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_nonpon_100x = read_vcf(f"{insilico_chm13_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_nonpon_100x = read_vcf(f"{insilico_chm13_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_nonpon_100x = read_vcf(f"{insilico_chm13_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n4_chm13_nonpon_100x = blt_t1n4_chm13_nonpon_100x[blt_t1n4_chm13_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_100x = blt_t1n9_chm13_nonpon_100x[blt_t1n9_chm13_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_100x = blt_t1n19_chm13_nonpon_100x[blt_t1n19_chm13_nonpon_100x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_chm13_nonpon_100x['SNVid_chm13'] = blt_t1n4_chm13_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_100x['SNVid_chm13'] = blt_t1n9_chm13_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_100x['SNVid_chm13'] = blt_t1n19_chm13_nonpon_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_100x['POSid'] = blt_t1n4_chm13_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_100x['POSid'] = blt_t1n9_chm13_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_100x['POSid'] = blt_t1n19_chm13_nonpon_100x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_nonpon_100x = blt_t1n4_chm13_nonpon_100x[~blt_t1n4_chm13_nonpon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_100x = blt_t1n9_chm13_nonpon_100x[~blt_t1n9_chm13_nonpon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_100x = blt_t1n19_chm13_nonpon_100x[~blt_t1n19_chm13_nonpon_100x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n4_chm13_nonpon_10x = read_vcf(f"{insilico_chm13_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n4_chm13_nonpon_20x = read_vcf(f"{insilico_chm13_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n4_chm13_nonpon_40x = read_vcf(f"{insilico_chm13_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n4_chm13_nonpon_10x = blt_t1n4_chm13_nonpon_10x[blt_t1n4_chm13_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_chm13_nonpon_20x = blt_t1n4_chm13_nonpon_20x[blt_t1n4_chm13_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n4_chm13_nonpon_40x = blt_t1n4_chm13_nonpon_40x[blt_t1n4_chm13_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_chm13_nonpon_10x['SNVid_chm13'] = blt_t1n4_chm13_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_20x['SNVid_chm13'] = blt_t1n4_chm13_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_40x['SNVid_chm13'] = blt_t1n4_chm13_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_10x['POSid'] = blt_t1n4_chm13_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_20x['POSid'] = blt_t1n4_chm13_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n4_chm13_nonpon_40x['POSid'] = blt_t1n4_chm13_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_nonpon_10x = blt_t1n4_chm13_nonpon_10x[~blt_t1n4_chm13_nonpon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n4_chm13_nonpon_20x = blt_t1n4_chm13_nonpon_20x[~blt_t1n4_chm13_nonpon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n4_chm13_nonpon_40x = blt_t1n4_chm13_nonpon_40x[~blt_t1n4_chm13_nonpon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n9_chm13_nonpon_10x = read_vcf(f"{insilico_chm13_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_nonpon_20x = read_vcf(f"{insilico_chm13_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_nonpon_40x = read_vcf(f"{insilico_chm13_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n9_chm13_nonpon_10x = blt_t1n9_chm13_nonpon_10x[blt_t1n9_chm13_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_20x = blt_t1n9_chm13_nonpon_20x[blt_t1n9_chm13_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_40x = blt_t1n9_chm13_nonpon_40x[blt_t1n9_chm13_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n9_chm13_nonpon_10x['SNVid_chm13'] = blt_t1n9_chm13_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_20x['SNVid_chm13'] = blt_t1n9_chm13_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_40x['SNVid_chm13'] = blt_t1n9_chm13_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_10x['POSid'] = blt_t1n9_chm13_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_20x['POSid'] = blt_t1n9_chm13_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_nonpon_40x['POSid'] = blt_t1n9_chm13_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n9_chm13_nonpon_10x = blt_t1n9_chm13_nonpon_10x[~blt_t1n9_chm13_nonpon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_20x = blt_t1n9_chm13_nonpon_20x[~blt_t1n9_chm13_nonpon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_nonpon_40x = blt_t1n9_chm13_nonpon_40x[~blt_t1n9_chm13_nonpon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

blt_t1n19_chm13_nonpon_10x = read_vcf(f"{insilico_chm13_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_nonpon_20x = read_vcf(f"{insilico_chm13_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_nonpon_40x = read_vcf(f"{insilico_chm13_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X_chm13_nonpon.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n19_chm13_nonpon_10x = blt_t1n19_chm13_nonpon_10x[blt_t1n19_chm13_nonpon_10x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_20x = blt_t1n19_chm13_nonpon_20x[blt_t1n19_chm13_nonpon_20x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_40x = blt_t1n19_chm13_nonpon_40x[blt_t1n19_chm13_nonpon_40x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n19_chm13_nonpon_10x['SNVid_chm13'] = blt_t1n19_chm13_nonpon_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_20x['SNVid_chm13'] = blt_t1n19_chm13_nonpon_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_40x['SNVid_chm13'] = blt_t1n19_chm13_nonpon_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_10x['POSid'] = blt_t1n19_chm13_nonpon_10x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_20x['POSid'] = blt_t1n19_chm13_nonpon_20x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_nonpon_40x['POSid'] = blt_t1n19_chm13_nonpon_40x[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n19_chm13_nonpon_10x = blt_t1n19_chm13_nonpon_10x[~blt_t1n19_chm13_nonpon_10x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_20x = blt_t1n19_chm13_nonpon_20x[~blt_t1n19_chm13_nonpon_20x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_nonpon_40x = blt_t1n19_chm13_nonpon_40x[~blt_t1n19_chm13_nonpon_40x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

# %%
# NOTE: Calculating Precision and Recall by SNVid_chm13 
# INFO: T1N4 100X
blt_t1n4_chm13_nonpon_100x_snv = set(blt_t1n4_chm13_nonpon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_nonpon_100x_snv_tp, blt_t1n4_chm13_nonpon_100x_snv_fp, blt_t1n4_chm13_nonpon_100x_snv_fn, blt_t1n4_chm13_nonpon_100x_snv_precision, blt_t1n4_chm13_nonpon_100x_snv_recall, blt_t1n4_chm13_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_nonpon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 40X
blt_t1n4_chm13_nonpon_40x_snv = set(blt_t1n4_chm13_nonpon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_nonpon_40x_snv_tp, blt_t1n4_chm13_nonpon_40x_snv_fp, blt_t1n4_chm13_nonpon_40x_snv_fn, blt_t1n4_chm13_nonpon_40x_snv_precision, blt_t1n4_chm13_nonpon_40x_snv_recall, blt_t1n4_chm13_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_nonpon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 20X
blt_t1n4_chm13_nonpon_20x_snv = set(blt_t1n4_chm13_nonpon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_nonpon_20x_snv_tp, blt_t1n4_chm13_nonpon_20x_snv_fp, blt_t1n4_chm13_nonpon_20x_snv_fn, blt_t1n4_chm13_nonpon_20x_snv_precision, blt_t1n4_chm13_nonpon_20x_snv_recall, blt_t1n4_chm13_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_nonpon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N4 10X
blt_t1n4_chm13_nonpon_10x_snv = set(blt_t1n4_chm13_nonpon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n4_chm13_nonpon_10x_snv_tp, blt_t1n4_chm13_nonpon_10x_snv_fp, blt_t1n4_chm13_nonpon_10x_snv_fn, blt_t1n4_chm13_nonpon_10x_snv_precision, blt_t1n4_chm13_nonpon_10x_snv_recall, blt_t1n4_chm13_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_nonpon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 100X
blt_t1n9_chm13_nonpon_100x_snv = set(blt_t1n9_chm13_nonpon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_nonpon_100x_snv_tp, blt_t1n9_chm13_nonpon_100x_snv_fp, blt_t1n9_chm13_nonpon_100x_snv_fn, blt_t1n9_chm13_nonpon_100x_snv_precision, blt_t1n9_chm13_nonpon_100x_snv_recall, blt_t1n9_chm13_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_nonpon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 40X
blt_t1n9_chm13_nonpon_40x_snv = set(blt_t1n9_chm13_nonpon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_nonpon_40x_snv_tp, blt_t1n9_chm13_nonpon_40x_snv_fp, blt_t1n9_chm13_nonpon_40x_snv_fn, blt_t1n9_chm13_nonpon_40x_snv_precision, blt_t1n9_chm13_nonpon_40x_snv_recall, blt_t1n9_chm13_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_nonpon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 20X
blt_t1n9_chm13_nonpon_20x_snv = set(blt_t1n9_chm13_nonpon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_nonpon_20x_snv_tp, blt_t1n9_chm13_nonpon_20x_snv_fp, blt_t1n9_chm13_nonpon_20x_snv_fn, blt_t1n9_chm13_nonpon_20x_snv_precision, blt_t1n9_chm13_nonpon_20x_snv_recall, blt_t1n9_chm13_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_nonpon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 10X
blt_t1n9_chm13_nonpon_10x_snv = set(blt_t1n9_chm13_nonpon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n9_chm13_nonpon_10x_snv_tp, blt_t1n9_chm13_nonpon_10x_snv_fp, blt_t1n9_chm13_nonpon_10x_snv_fn, blt_t1n9_chm13_nonpon_10x_snv_precision, blt_t1n9_chm13_nonpon_10x_snv_recall, blt_t1n9_chm13_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_nonpon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 100X
blt_t1n19_chm13_nonpon_100x_snv = set(blt_t1n19_chm13_nonpon_100x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_nonpon_100x_snv_tp, blt_t1n19_chm13_nonpon_100x_snv_fp, blt_t1n19_chm13_nonpon_100x_snv_fn, blt_t1n19_chm13_nonpon_100x_snv_precision, blt_t1n19_chm13_nonpon_100x_snv_recall, blt_t1n19_chm13_nonpon_100x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_nonpon_100x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 40X
blt_t1n19_chm13_nonpon_40x_snv = set(blt_t1n19_chm13_nonpon_40x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_nonpon_40x_snv_tp, blt_t1n19_chm13_nonpon_40x_snv_fp, blt_t1n19_chm13_nonpon_40x_snv_fn, blt_t1n19_chm13_nonpon_40x_snv_precision, blt_t1n19_chm13_nonpon_40x_snv_recall, blt_t1n19_chm13_nonpon_40x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_nonpon_40x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 20X
blt_t1n19_chm13_nonpon_20x_snv = set(blt_t1n19_chm13_nonpon_20x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_nonpon_20x_snv_tp, blt_t1n19_chm13_nonpon_20x_snv_fp, blt_t1n19_chm13_nonpon_20x_snv_fn, blt_t1n19_chm13_nonpon_20x_snv_precision, blt_t1n19_chm13_nonpon_20x_snv_recall, blt_t1n19_chm13_nonpon_20x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_nonpon_20x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 10X
blt_t1n19_chm13_nonpon_10x_snv = set(blt_t1n19_chm13_nonpon_10x["SNVid_chm13"].values) - set(bl_100x_chm13_nonpon_snv["SNVid_chm13"].values)
blt_t1n19_chm13_nonpon_10x_snv_tp, blt_t1n19_chm13_nonpon_10x_snv_fp, blt_t1n19_chm13_nonpon_10x_snv_fn, blt_t1n19_chm13_nonpon_10x_snv_precision, blt_t1n19_chm13_nonpon_10x_snv_recall, blt_t1n19_chm13_nonpon_10x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_nonpon_10x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# %%
# ADDED: Illumina short-read (Diploid 100X, haploid 50X) for evaluating mSNV discovery performance - T2T-CHM13
# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures for T2T-CHM13
platform="Illumina"
bl_sr_chm13_50x_snv = read_vcf(f"{insilico_chm13_dir}/N_ONLY/{platform}/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_sr_50X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

bl_sr_chm13_50x_snv = bl_sr_chm13_50x_snv[bl_sr_chm13_50x_snv["CHROM"].isin(primary_chrs)].reset_index(drop=True)
bl_sr_chm13_50x_snv['SNVid_chm13'] = bl_sr_chm13_50x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_sr_chm13_50x_snv['POSid'] = bl_sr_chm13_50x_snv[['CHROM', 'POS']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_sr_50x = read_vcf(f"{insilico_chm13_dir}/T1N4/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_sr_50X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n9_chm13_sr_50x = read_vcf(f"{insilico_chm13_dir}/T1N9/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_sr_50X_chm13.deepsomatictonly.PASS.snv.vcf.gz")
blt_t1n19_chm13_sr_50x = read_vcf(f"{insilico_chm13_dir}/T1N19/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_sr_50X_chm13.deepsomatictonly.PASS.snv.vcf.gz")

blt_t1n4_chm13_sr_50x = blt_t1n4_chm13_sr_50x[blt_t1n4_chm13_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n9_chm13_sr_50x = blt_t1n9_chm13_sr_50x[blt_t1n9_chm13_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)
blt_t1n19_chm13_sr_50x = blt_t1n19_chm13_sr_50x[blt_t1n19_chm13_sr_50x["CHROM"].isin(primary_chrs)].reset_index(drop=True)

blt_t1n4_chm13_sr_50x['SNVid_chm13'] = blt_t1n4_chm13_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_sr_50x['SNVid_chm13'] = blt_t1n9_chm13_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_sr_50x['SNVid_chm13'] = blt_t1n19_chm13_sr_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_sr_50x["POSid"] = blt_t1n4_chm13_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_chm13_sr_50x["POSid"] = blt_t1n9_chm13_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_chm13_sr_50x["POSid"] = blt_t1n19_chm13_sr_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_chm13_sr_50x = blt_t1n4_chm13_sr_50x[~blt_t1n4_chm13_sr_50x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n9_chm13_sr_50x = blt_t1n9_chm13_sr_50x[~blt_t1n9_chm13_sr_50x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)
blt_t1n19_chm13_sr_50x = blt_t1n19_chm13_sr_50x[~blt_t1n19_chm13_sr_50x["POSid"].isin(snv_flagset_chm13_position_set)].reset_index(drop=True)

# INFO: Calculating Precision and Recall in the DSA space but FOR SHORT-READ (Illumina) sequencing data
# INFO: T1N4 50X
blt_t1n4_chm13_sr_50x_snv = set(blt_t1n4_chm13_sr_50x["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv["SNVid_chm13"].values)
blt_t1n4_chm13_sr_50x_snv_tp, blt_t1n4_chm13_sr_50x_snv_fp, blt_t1n4_chm13_sr_50x_snv_fn, blt_t1n4_chm13_sr_50x_snv_precision, blt_t1n4_chm13_sr_50x_snv_recall, blt_t1n4_chm13_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_sr_50x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N9 50X
blt_t1n9_chm13_sr_50x_snv = set(blt_t1n9_chm13_sr_50x["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv["SNVid_chm13"].values)
blt_t1n9_chm13_sr_50x_snv_tp, blt_t1n9_chm13_sr_50x_snv_fp, blt_t1n9_chm13_sr_50x_snv_fn, blt_t1n9_chm13_sr_50x_snv_precision, blt_t1n9_chm13_sr_50x_snv_recall, blt_t1n9_chm13_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_sr_50x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)


# INFO: T1N19 50X
blt_t1n19_chm13_sr_50x_snv = set(blt_t1n19_chm13_sr_50x["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv["SNVid_chm13"].values)
blt_t1n19_chm13_sr_50x_snv_tp, blt_t1n19_chm13_sr_50x_snv_fp, blt_t1n19_chm13_sr_50x_snv_fn, blt_t1n19_chm13_sr_50x_snv_precision, blt_t1n19_chm13_sr_50x_snv_recall, blt_t1n19_chm13_sr_50x_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_sr_50x_snv, snv_referenceset_chm13_primary_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13)

# %%
# ADDED: Non-Satellite Regions of the T2T-CHM13
# INFO: PacBio
bl_100x_chm13_pon_snv_non_satellite = vcf_in_pyranges_interval(bl_100x_chm13_pon_snv, chm13_non_satellite_pr, id="SNVid_chm13") 

blt_t1n4_chm13_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_chm13_pon_20x, chm13_non_satellite_pr, id="SNVid_chm13")
blt_t1n9_chm13_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_chm13_pon_20x, chm13_non_satellite_pr, id="SNVid_chm13")
blt_t1n19_chm13_pon_20x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_chm13_pon_20x, chm13_non_satellite_pr, id="SNVid_chm13")

# INFO: T1N4 20X
blt_t1n4_chm13_pon_20x_non_satellite_snv = set(blt_t1n4_chm13_pon_20x_non_satellite["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv_non_satellite["SNVid_chm13"].values)
blt_t1n4_chm13_pon_20x_non_satellite_snv_tp, blt_t1n4_chm13_pon_20x_non_satellite_snv_fp, blt_t1n4_chm13_pon_20x_non_satellite_snv_fn, blt_t1n4_chm13_pon_20x_non_satellite_snv_precision, blt_t1n4_chm13_pon_20x_non_satellite_snv_recall, blt_t1n4_chm13_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_pon_20x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# INFO: T1N9 20X
blt_t1n9_chm13_pon_20x_non_satellite_snv = set(blt_t1n9_chm13_pon_20x_non_satellite["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv_non_satellite["SNVid_chm13"].values)
blt_t1n9_chm13_pon_20x_non_satellite_snv_tp, blt_t1n9_chm13_pon_20x_non_satellite_snv_fp, blt_t1n9_chm13_pon_20x_non_satellite_snv_fn, blt_t1n9_chm13_pon_20x_non_satellite_snv_precision, blt_t1n9_chm13_pon_20x_non_satellite_snv_recall, blt_t1n9_chm13_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_pon_20x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# INFO: T1N19 20X
blt_t1n19_chm13_pon_20x_non_satellite_snv = set(blt_t1n19_chm13_pon_20x_non_satellite["SNVid_chm13"].values) - set(bl_100x_chm13_pon_snv_non_satellite["SNVid_chm13"].values)
blt_t1n19_chm13_pon_20x_non_satellite_snv_tp, blt_t1n19_chm13_pon_20x_non_satellite_snv_fp, blt_t1n19_chm13_pon_20x_non_satellite_snv_fn, blt_t1n19_chm13_pon_20x_non_satellite_snv_precision, blt_t1n19_chm13_pon_20x_non_satellite_snv_recall, blt_t1n19_chm13_pon_20x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_pon_20x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# INFO: Illumina Short-read
bl_sr_chm13_50x_snv_non_satellite = vcf_in_pyranges_interval(bl_sr_chm13_50x_snv, chm13_non_satellite_pr, id="SNVid_chm13")

blt_t1n4_chm13_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n4_chm13_sr_50x, chm13_non_satellite_pr, id="SNVid_chm13")
blt_t1n9_chm13_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n9_chm13_sr_50x, chm13_non_satellite_pr, id="SNVid_chm13")
blt_t1n19_chm13_sr_50x_non_satellite = vcf_in_pyranges_interval(blt_t1n19_chm13_sr_50x, chm13_non_satellite_pr, id="SNVid_chm13")

# INFO: T1N4 50X
blt_t1n4_chm13_sr_50x_non_satellite_snv = set(blt_t1n4_chm13_sr_50x_non_satellite["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv_non_satellite["SNVid_chm13"].values)
blt_t1n4_chm13_sr_50x_non_satellite_snv_tp, blt_t1n4_chm13_sr_50x_non_satellite_snv_fp, blt_t1n4_chm13_sr_50x_non_satellite_snv_fn, blt_t1n4_chm13_sr_50x_non_satellite_snv_precision, blt_t1n4_chm13_sr_50x_non_satellite_snv_recall, blt_t1n4_chm13_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n4_chm13_sr_50x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# INFO: T1N9 50X
blt_t1n9_chm13_sr_50x_non_satellite_snv = set(blt_t1n9_chm13_sr_50x_non_satellite["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv_non_satellite["SNVid_chm13"].values)
blt_t1n9_chm13_sr_50x_non_satellite_snv_tp, blt_t1n9_chm13_sr_50x_non_satellite_snv_fp, blt_t1n9_chm13_sr_50x_non_satellite_snv_fn, blt_t1n9_chm13_sr_50x_non_satellite_snv_precision, blt_t1n9_chm13_sr_50x_non_satellite_snv_recall, blt_t1n9_chm13_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n9_chm13_sr_50x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# INFO: T1N19 50X
blt_t1n19_chm13_sr_50x_non_satellite_snv = set(blt_t1n19_chm13_sr_50x_non_satellite["SNVid_chm13"].values) - set(bl_sr_chm13_50x_snv_non_satellite["SNVid_chm13"].values)
blt_t1n19_chm13_sr_50x_non_satellite_snv_tp, blt_t1n19_chm13_sr_50x_non_satellite_snv_fp, blt_t1n19_chm13_sr_50x_non_satellite_snv_fn, blt_t1n19_chm13_sr_50x_non_satellite_snv_precision, blt_t1n19_chm13_sr_50x_non_satellite_snv_recall, blt_t1n19_chm13_sr_50x_non_satellite_snv_recall_withreject = snv_pr_metrics(blt_t1n19_chm13_sr_50x_non_satellite_snv, snv_referenceset_chm13_primary_non_satellite_set, extra_fn=n_dsa_snv_nonsurjected_to_chm13_satellite)

# %%
# INFO: Analysis in DSA-space Using DSA-based SNVs that could be surjected to T2T-CHM13
# NOTE: Defining new set of DSA-based reference set is needed
snv_referenceset_pgfbsnvid_snvid_tab['SNVid_chm13'] = snv_referenceset_pgfbsnvid_snvid_tab['SNVid'].map(dict(zip(snv_referenceset_chm13_position['SNVid'], snv_referenceset_chm13_position['SNVid_chm13'])))

snv_flagset_position_chm13_compatible_set = snv_flagset_position_set.union(set(snv_referenceset_pgfbsnvid_snvid_tab[~(snv_referenceset_pgfbsnvid_snvid_tab["SNVid_chm13"].isin(snv_referenceset_chm13_primary_set))]["SNVid"].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").values)) # NOTE: Flagged SNV positions with added DSA-only SNVs that could not be surjected to hg38

snv_referenceset_chm13_compatible = set(snv_referenceset_pgfbsnvid_snvid_tab[~(snv_referenceset_pgfbsnvid_snvid_tab["SNVid"].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").isin(snv_flagset_position_chm13_compatible_set))]["SNVid"].values)


# %%
# INFO: Analysis in DSA-space Using DSA-based SNVs that could NOT!! be surjected to T2T-CHM13 (DSA-only SNVs in temrs of CHM13)
# NOTE: Defining new set of DSA-based reference set is needed
colotb_shared_snv_dsaonly_chm13 = vcf_in_pyranges_interval(colotb_shared_snv, dsa_only_chm13_pr_1kb)

snv_referenceset_dsa_only_chm13_set = set(colotb_shared_snv_dsaonly_chm13["SNVid"].values)

snv_flagset_position_set_plus_chm13_surjectable_position = snv_flagset_position_set.union(set(map(lambda x: '_'.join(x.split('_')[0:2]), (snv_referenceset - snv_referenceset_dsa_only_chm13_set)))) # NOTE: Flagged positions + DSA-based SNVs that could be surjected to T2T-CHM13 (Also flagged)

# %%
bl_100x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_100x_snv['SNVid'] = bl_100x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_snv = bl_100x_snv[(vcf_info_getter(bl_100x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_100x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_100x_snv = vcf_in_pyranges_interval(bl_100x_snv, callable_pr)
bl_100x_snv = vcf_in_pyranges_interval(bl_100x_snv, dsa_only_chm13_pr_1kb)

# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures
blt_t1n4_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_dsaonly_100x = blt_t1n4_dsaonly_100x[(vcf_info_getter(blt_t1n4_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_100x = blt_t1n9_dsaonly_100x[(vcf_info_getter(blt_t1n9_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_100x = blt_t1n19_dsaonly_100x[(vcf_info_getter(blt_t1n19_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_dsaonly_100x['SNVid'] = blt_t1n4_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_100x['SNVid'] = blt_t1n9_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_100x['SNVid'] = blt_t1n19_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_100x["POSid"] = blt_t1n4_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_100x["POSid"] = blt_t1n9_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_100x["POSid"] = blt_t1n19_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_100x, callable_pr)
blt_t1n4_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_100x, dsa_only_chm13_pr_1kb)
blt_t1n9_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_100x, callable_pr)
blt_t1n9_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_100x, dsa_only_chm13_pr_1kb)
blt_t1n19_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_100x, callable_pr)
blt_t1n19_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_100x, dsa_only_chm13_pr_1kb)

blt_t1n4_dsaonly_100x = blt_t1n4_dsaonly_100x[~blt_t1n4_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_100x = blt_t1n9_dsaonly_100x[~blt_t1n9_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_100x = blt_t1n19_dsaonly_100x[~blt_t1n19_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)

blt_t1n4_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_dsaonly_10x = blt_t1n4_dsaonly_10x[(vcf_info_getter(blt_t1n4_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_dsaonly_20x = blt_t1n4_dsaonly_20x[(vcf_info_getter(blt_t1n4_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_dsaonly_40x = blt_t1n4_dsaonly_40x[(vcf_info_getter(blt_t1n4_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_dsaonly_10x['SNVid'] = blt_t1n4_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_20x['SNVid'] = blt_t1n4_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_40x['SNVid'] = blt_t1n4_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_10x["POSid"] = blt_t1n4_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_20x["POSid"] = blt_t1n4_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_40x["POSid"] = blt_t1n4_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_10x, callable_pr)
blt_t1n4_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_10x, dsa_only_chm13_pr_1kb)
blt_t1n4_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_20x, callable_pr)
blt_t1n4_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_20x, dsa_only_chm13_pr_1kb)
blt_t1n4_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_40x, callable_pr)
blt_t1n4_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_40x, dsa_only_chm13_pr_1kb)

blt_t1n4_dsaonly_10x = blt_t1n4_dsaonly_10x[~blt_t1n4_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n4_dsaonly_20x = blt_t1n4_dsaonly_20x[~blt_t1n4_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n4_dsaonly_40x = blt_t1n4_dsaonly_40x[~blt_t1n4_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)

blt_t1n9_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n9_dsaonly_10x = blt_t1n9_dsaonly_10x[(vcf_info_getter(blt_t1n9_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_20x = blt_t1n9_dsaonly_20x[(vcf_info_getter(blt_t1n9_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_40x = blt_t1n9_dsaonly_40x[(vcf_info_getter(blt_t1n9_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n9_dsaonly_10x['SNVid'] = blt_t1n9_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_20x['SNVid'] = blt_t1n9_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_40x['SNVid'] = blt_t1n9_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n9_dsaonly_10x["POSid"] = blt_t1n9_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_20x["POSid"] = blt_t1n9_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_40x["POSid"] = blt_t1n9_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n9_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_10x, callable_pr)
blt_t1n9_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_10x, dsa_only_chm13_pr_1kb)
blt_t1n9_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_20x, callable_pr)
blt_t1n9_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_20x, dsa_only_chm13_pr_1kb)
blt_t1n9_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_40x, callable_pr)
blt_t1n9_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_40x, dsa_only_chm13_pr_1kb)

blt_t1n9_dsaonly_10x = blt_t1n9_dsaonly_10x[~blt_t1n9_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_20x = blt_t1n9_dsaonly_20x[~blt_t1n9_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_40x = blt_t1n9_dsaonly_40x[~blt_t1n9_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)

blt_t1n19_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n19_dsaonly_10x = blt_t1n19_dsaonly_10x[(vcf_info_getter(blt_t1n19_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_20x = blt_t1n19_dsaonly_20x[(vcf_info_getter(blt_t1n19_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_40x = blt_t1n19_dsaonly_40x[(vcf_info_getter(blt_t1n19_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n19_dsaonly_10x['SNVid'] = blt_t1n19_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_20x['SNVid'] = blt_t1n19_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_40x['SNVid'] = blt_t1n19_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n19_dsaonly_10x["POSid"] = blt_t1n19_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_20x["POSid"] = blt_t1n19_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_40x["POSid"] = blt_t1n19_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n19_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_10x, callable_pr)
blt_t1n19_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_10x, dsa_only_chm13_pr_1kb)
blt_t1n19_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_20x, callable_pr)
blt_t1n19_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_20x, dsa_only_chm13_pr_1kb)
blt_t1n19_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_40x, callable_pr)
blt_t1n19_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_40x, dsa_only_chm13_pr_1kb)

blt_t1n19_dsaonly_10x = blt_t1n19_dsaonly_10x[~blt_t1n19_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_20x = blt_t1n19_dsaonly_20x[~blt_t1n19_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_40x = blt_t1n19_dsaonly_40x[~blt_t1n19_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)

# %%
# INFO: T1N4 100X
blt_t1n4_100x_dsa_only_chm13_snv = set(blt_t1n4_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_100x_dsa_only_chm13_snv_tp, blt_t1n4_100x_dsa_only_chm13_snv_fp, blt_t1n4_100x_dsa_only_chm13_snv_fn, blt_t1n4_100x_dsa_only_chm13_snv_precision, blt_t1n4_100x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n4_100x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N4 40X
blt_t1n4_40x_dsa_only_chm13_snv = set(blt_t1n4_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_40x_dsa_only_chm13_snv_tp, blt_t1n4_40x_dsa_only_chm13_snv_fp, blt_t1n4_40x_dsa_only_chm13_snv_fn, blt_t1n4_40x_dsa_only_chm13_snv_precision, blt_t1n4_40x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n4_40x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N4 20X
blt_t1n4_20x_dsa_only_chm13_snv = set(blt_t1n4_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_20x_dsa_only_chm13_snv_tp, blt_t1n4_20x_dsa_only_chm13_snv_fp, blt_t1n4_20x_dsa_only_chm13_snv_fn, blt_t1n4_20x_dsa_only_chm13_snv_precision, blt_t1n4_20x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n4_20x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N4 10X
blt_t1n4_10x_dsa_only_chm13_snv = set(blt_t1n4_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_10x_dsa_only_chm13_snv_tp, blt_t1n4_10x_dsa_only_chm13_snv_fp, blt_t1n4_10x_dsa_only_chm13_snv_fn, blt_t1n4_10x_dsa_only_chm13_snv_precision, blt_t1n4_10x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n4_10x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N9 100X
blt_t1n9_100x_dsa_only_chm13_snv = set(blt_t1n9_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_100x_dsa_only_chm13_snv_tp, blt_t1n9_100x_dsa_only_chm13_snv_fp, blt_t1n9_100x_dsa_only_chm13_snv_fn, blt_t1n9_100x_dsa_only_chm13_snv_precision, blt_t1n9_100x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n9_100x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N9 40X
blt_t1n9_40x_dsa_only_chm13_snv = set(blt_t1n9_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_40x_dsa_only_chm13_snv_tp, blt_t1n9_40x_dsa_only_chm13_snv_fp, blt_t1n9_40x_dsa_only_chm13_snv_fn, blt_t1n9_40x_dsa_only_chm13_snv_precision, blt_t1n9_40x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n9_40x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N9 20X
blt_t1n9_20x_dsa_only_chm13_snv = set(blt_t1n9_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n9_20x_dsa_only_chm13_snv_tp, blt_t1n9_20x_dsa_only_chm13_snv_fp, blt_t1n9_20x_dsa_only_chm13_snv_fn, blt_t1n9_20x_dsa_only_chm13_snv_precision, blt_t1n9_20x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n9_20x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N9 10X
blt_t1n9_10x_dsa_only_chm13_snv = set(blt_t1n9_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_10x_dsa_only_chm13_snv_tp, blt_t1n9_10x_dsa_only_chm13_snv_fp, blt_t1n9_10x_dsa_only_chm13_snv_fn, blt_t1n9_10x_dsa_only_chm13_snv_precision, blt_t1n9_10x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n9_10x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N19 100X
blt_t1n19_100x_dsa_only_chm13_snv = set(blt_t1n19_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n19_100x_dsa_only_chm13_snv_tp, blt_t1n19_100x_dsa_only_chm13_snv_fp, blt_t1n19_100x_dsa_only_chm13_snv_fn, blt_t1n19_100x_dsa_only_chm13_snv_precision, blt_t1n19_100x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n19_100x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N19 40X
blt_t1n19_40x_dsa_only_chm13_snv = set(blt_t1n19_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_40x_dsa_only_chm13_snv_tp, blt_t1n19_40x_dsa_only_chm13_snv_fp, blt_t1n19_40x_dsa_only_chm13_snv_fn, blt_t1n19_40x_dsa_only_chm13_snv_precision, blt_t1n19_40x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n19_40x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N19 20X
blt_t1n19_20x_dsa_only_chm13_snv = set(blt_t1n19_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_20x_dsa_only_chm13_snv_tp, blt_t1n19_20x_dsa_only_chm13_snv_fp, blt_t1n19_20x_dsa_only_chm13_snv_fn, blt_t1n19_20x_dsa_only_chm13_snv_precision, blt_t1n19_20x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n19_20x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N19 10X
blt_t1n19_10x_dsa_only_chm13_snv = set(blt_t1n19_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_10x_dsa_only_chm13_snv_tp, blt_t1n19_10x_dsa_only_chm13_snv_fp, blt_t1n19_10x_dsa_only_chm13_snv_fn, blt_t1n19_10x_dsa_only_chm13_snv_precision, blt_t1n19_10x_dsa_only_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n19_10x_dsa_only_chm13_snv, snv_referenceset_dsa_only_chm13_set)



# %%
# ADDED: Illumina short-read (Diploid 100X, haploid 50X) for evaluating mSNV discovery performance
# INFO: Analysis in DSA-space Using short-read DSA-based SNVs that could NOT!! be surjected to T2T-CHM13 (DSA-only SNVs in terms of CHM13)
platform="Illumina"
bl_sr_50x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/{platform}/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_sr_50x_snv['SNVid'] = bl_sr_50x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_sr_50x_snv = bl_sr_50x_snv[(vcf_info_getter(bl_sr_50x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_sr_50x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_sr_50x_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, callable_pr)
bl_sr_50x_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, dsa_only_chm13_pr_1kb)

blt_t1n4_sr_dsaonly_chm13_50x = read_vcf(f"{insilico_dsa_dir}/T1N4/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_sr_dsaonly_chm13_50x = read_vcf(f"{insilico_dsa_dir}/T1N9/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_sr_dsaonly_chm13_50x = read_vcf(f"{insilico_dsa_dir}/T1N19/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_sr_dsaonly_chm13_50x = blt_t1n4_sr_dsaonly_chm13_50x[(vcf_info_getter(blt_t1n4_sr_dsaonly_chm13_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_sr_dsaonly_chm13_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_sr_dsaonly_chm13_50x = blt_t1n9_sr_dsaonly_chm13_50x[(vcf_info_getter(blt_t1n9_sr_dsaonly_chm13_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_sr_dsaonly_chm13_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_sr_dsaonly_chm13_50x = blt_t1n19_sr_dsaonly_chm13_50x[(vcf_info_getter(blt_t1n19_sr_dsaonly_chm13_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_sr_dsaonly_chm13_50x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_sr_dsaonly_chm13_50x['SNVid'] = blt_t1n4_sr_dsaonly_chm13_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_dsaonly_chm13_50x['SNVid'] = blt_t1n9_sr_dsaonly_chm13_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_dsaonly_chm13_50x['SNVid'] = blt_t1n19_sr_dsaonly_chm13_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_dsaonly_chm13_50x["POSid"] = blt_t1n4_sr_dsaonly_chm13_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_dsaonly_chm13_50x["POSid"] = blt_t1n9_sr_dsaonly_chm13_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_dsaonly_chm13_50x["POSid"] = blt_t1n19_sr_dsaonly_chm13_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n4_sr_dsaonly_chm13_50x, callable_pr)
blt_t1n4_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n4_sr_dsaonly_chm13_50x, dsa_only_chm13_pr_1kb)
blt_t1n9_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n9_sr_dsaonly_chm13_50x, callable_pr)
blt_t1n9_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n9_sr_dsaonly_chm13_50x, dsa_only_chm13_pr_1kb)
blt_t1n19_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n19_sr_dsaonly_chm13_50x, callable_pr)
blt_t1n19_sr_dsaonly_chm13_50x = vcf_in_pyranges_interval(blt_t1n19_sr_dsaonly_chm13_50x, dsa_only_chm13_pr_1kb)

blt_t1n4_sr_dsaonly_chm13_50x = blt_t1n4_sr_dsaonly_chm13_50x[~blt_t1n4_sr_dsaonly_chm13_50x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n9_sr_dsaonly_chm13_50x = blt_t1n9_sr_dsaonly_chm13_50x[~blt_t1n9_sr_dsaonly_chm13_50x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)
blt_t1n19_sr_dsaonly_chm13_50x = blt_t1n19_sr_dsaonly_chm13_50x[~blt_t1n19_sr_dsaonly_chm13_50x["POSid"].isin(snv_flagset_position_set_plus_chm13_surjectable_position)].reset_index(drop=True)

# INFO: T1N4 50X
blt_t1n4_sr_dsaonly_50X_chm13_snv = set(blt_t1n4_sr_dsaonly_chm13_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n4_sr_dsaonly_50X_chm13_snv_tp, blt_t1n4_sr_dsaonly_50X_chm13_snv_fp, blt_t1n4_sr_dsaonly_50X_chm13_snv_fn, blt_t1n4_sr_dsaonly_50X_chm13_snv_precision, blt_t1n4_sr_dsaonly_50X_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n4_sr_dsaonly_50X_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N9 50X
blt_t1n9_sr_dsaonly_50X_chm13_snv = set(blt_t1n9_sr_dsaonly_chm13_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n9_sr_dsaonly_50X_chm13_snv_tp, blt_t1n9_sr_dsaonly_50X_chm13_snv_fp, blt_t1n9_sr_dsaonly_50X_chm13_snv_fn, blt_t1n9_sr_dsaonly_50X_chm13_snv_precision, blt_t1n9_sr_dsaonly_50X_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n9_sr_dsaonly_50X_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: T1N19 50X
blt_t1n19_sr_dsaonly_50X_chm13_snv = set(blt_t1n19_sr_dsaonly_chm13_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n19_sr_dsaonly_50X_chm13_snv_tp, blt_t1n19_sr_dsaonly_50X_chm13_snv_fp, blt_t1n19_sr_dsaonly_50X_chm13_snv_fn, blt_t1n19_sr_dsaonly_50X_chm13_snv_precision, blt_t1n19_sr_dsaonly_50X_chm13_snv_recall, _ = snv_pr_metrics(blt_t1n19_sr_dsaonly_50X_chm13_snv, snv_referenceset_dsa_only_chm13_set)


# INFO: Using `snv_referenceset_between_hap_surjected` to calculate Precision on DSA-only SNVs compared to CHM13
# INFO: T1N4 50X
blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_tp, blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_fp, blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_fn, blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_precision, blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n4_sr_dsaonly_50X_chm13_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N9 50X
blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_tp, blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_fp, blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_fn, blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_precision, blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n9_sr_dsaonly_50X_chm13_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N19 50X
blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_tp, blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_fp, blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_fn, blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_precision, blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n19_sr_dsaonly_50X_chm13_snv, snv_referenceset_between_hap_surjected)

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################

# %%
# INFO: Analysis in DSA-space Using DSA-based SNVs that could be surjected to GRCh38
# NOTE: Defining new set of DSA-based reference set is needed
snv_referenceset_pgfbsnvid_snvid_tab['SNVid_hg38'] = snv_referenceset_pgfbsnvid_snvid_tab['SNVid'].map(dict(zip(snv_referenceset_hg38_position['SNVid'], snv_referenceset_hg38_position['SNVid_hg38'])))

snv_flagset_position_hg38_compatible_set = snv_flagset_position_set.union(set(snv_referenceset_pgfbsnvid_snvid_tab[~(snv_referenceset_pgfbsnvid_snvid_tab["SNVid_hg38"].isin(snv_referenceset_hg38_primary_set))]["SNVid"].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").values)) # NOTE: Flagged SNV positions with added DSA-only SNVs that could not be surjected to hg38

snv_referenceset_hg38_compatible = set(snv_referenceset_pgfbsnvid_snvid_tab[~(snv_referenceset_pgfbsnvid_snvid_tab["SNVid"].str.split('_').apply(lambda x: f"{x[0]}_{x[1]}").isin(snv_flagset_position_hg38_compatible_set))]["SNVid"].values)


# %%
# INFO: Analysis in DSA-space Using DSA-based SNVs that could NOT!! be surjected to GRCh38 (DSA-only SNVs)
# NOTE: Defining new set of DSA-based reference set is needed
colotb_shared_snv_dsaonly = vcf_in_pyranges_interval(colotb_shared_snv, dsa_only_pr_1kb)

snv_referenceset_dsa_only_set = set(colotb_shared_snv_dsaonly["SNVid"].values)

snv_flagset_position_set_plus_hg38_surjectable_position = snv_flagset_position_set.union(set(map(lambda x: '_'.join(x.split('_')[0:2]), (snv_referenceset - snv_referenceset_dsa_only_set)))) # NOTE: Flagged positions + DSA-based SNVs that could be surjected to GRCh38 (Also flagged)


# %%
bl_100x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_100x_snv['SNVid'] = bl_100x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_100x_snv = bl_100x_snv[(vcf_info_getter(bl_100x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_100x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_100x_snv = vcf_in_pyranges_interval(bl_100x_snv, callable_pr)
bl_100x_snv = vcf_in_pyranges_interval(bl_100x_snv, dsa_only_pr_1kb)

# INFO: DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures
blt_t1n4_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N4/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N9/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_100x = read_vcf(f"{insilico_dsa_dir}/T1N19/100X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_100X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_dsaonly_100x = blt_t1n4_dsaonly_100x[(vcf_info_getter(blt_t1n4_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_100x = blt_t1n9_dsaonly_100x[(vcf_info_getter(blt_t1n9_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_100x = blt_t1n19_dsaonly_100x[(vcf_info_getter(blt_t1n19_dsaonly_100x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_100x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_dsaonly_100x['SNVid'] = blt_t1n4_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_100x['SNVid'] = blt_t1n9_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_100x['SNVid'] = blt_t1n19_dsaonly_100x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_100x["POSid"] = blt_t1n4_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_100x["POSid"] = blt_t1n9_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_100x["POSid"] = blt_t1n19_dsaonly_100x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_100x, callable_pr)
blt_t1n4_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_100x, dsa_only_pr_1kb)
blt_t1n9_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_100x, callable_pr)
blt_t1n9_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_100x, dsa_only_pr_1kb)
blt_t1n19_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_100x, callable_pr)
blt_t1n19_dsaonly_100x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_100x, dsa_only_pr_1kb)

blt_t1n4_dsaonly_100x = blt_t1n4_dsaonly_100x[~blt_t1n4_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_100x = blt_t1n9_dsaonly_100x[~blt_t1n9_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_100x = blt_t1n19_dsaonly_100x[~blt_t1n19_dsaonly_100x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)

blt_t1n4_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N4/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N4/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n4_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N4/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_dsaonly_10x = blt_t1n4_dsaonly_10x[(vcf_info_getter(blt_t1n4_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_dsaonly_20x = blt_t1n4_dsaonly_20x[(vcf_info_getter(blt_t1n4_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n4_dsaonly_40x = blt_t1n4_dsaonly_40x[(vcf_info_getter(blt_t1n4_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_dsaonly_10x['SNVid'] = blt_t1n4_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_20x['SNVid'] = blt_t1n4_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_40x['SNVid'] = blt_t1n4_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_10x["POSid"] = blt_t1n4_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_20x["POSid"] = blt_t1n4_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n4_dsaonly_40x["POSid"] = blt_t1n4_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_10x, callable_pr)
blt_t1n4_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_10x, dsa_only_pr_1kb)
blt_t1n4_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_20x, callable_pr)
blt_t1n4_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_20x, dsa_only_pr_1kb)
blt_t1n4_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_40x, callable_pr)
blt_t1n4_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n4_dsaonly_40x, dsa_only_pr_1kb)

blt_t1n4_dsaonly_10x = blt_t1n4_dsaonly_10x[~blt_t1n4_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n4_dsaonly_20x = blt_t1n4_dsaonly_20x[~blt_t1n4_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n4_dsaonly_40x = blt_t1n4_dsaonly_40x[~blt_t1n4_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)

blt_t1n9_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N9/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N9/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N9/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n9_dsaonly_10x = blt_t1n9_dsaonly_10x[(vcf_info_getter(blt_t1n9_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_20x = blt_t1n9_dsaonly_20x[(vcf_info_getter(blt_t1n9_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_dsaonly_40x = blt_t1n9_dsaonly_40x[(vcf_info_getter(blt_t1n9_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n9_dsaonly_10x['SNVid'] = blt_t1n9_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_20x['SNVid'] = blt_t1n9_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_40x['SNVid'] = blt_t1n9_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n9_dsaonly_10x["POSid"] = blt_t1n9_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_20x["POSid"] = blt_t1n9_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_dsaonly_40x["POSid"] = blt_t1n9_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n9_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_10x, callable_pr)
blt_t1n9_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_10x, dsa_only_pr_1kb)
blt_t1n9_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_20x, callable_pr)
blt_t1n9_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_20x, dsa_only_pr_1kb)
blt_t1n9_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_40x, callable_pr)
blt_t1n9_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n9_dsaonly_40x, dsa_only_pr_1kb)

blt_t1n9_dsaonly_10x = blt_t1n9_dsaonly_10x[~blt_t1n9_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_20x = blt_t1n9_dsaonly_20x[~blt_t1n9_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n9_dsaonly_40x = blt_t1n9_dsaonly_40x[~blt_t1n9_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)

blt_t1n19_dsaonly_10x = read_vcf(f"{insilico_dsa_dir}/T1N19/10X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_10X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_20x = read_vcf(f"{insilico_dsa_dir}/T1N19/20X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_20X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_dsaonly_40x = read_vcf(f"{insilico_dsa_dir}/T1N19/40X/Fiber-seq/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_40X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n19_dsaonly_10x = blt_t1n19_dsaonly_10x[(vcf_info_getter(blt_t1n19_dsaonly_10x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_10x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_20x = blt_t1n19_dsaonly_20x[(vcf_info_getter(blt_t1n19_dsaonly_20x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_20x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_dsaonly_40x = blt_t1n19_dsaonly_40x[(vcf_info_getter(blt_t1n19_dsaonly_40x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_dsaonly_40x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n19_dsaonly_10x['SNVid'] = blt_t1n19_dsaonly_10x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_20x['SNVid'] = blt_t1n19_dsaonly_20x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_40x['SNVid'] = blt_t1n19_dsaonly_40x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n19_dsaonly_10x["POSid"] = blt_t1n19_dsaonly_10x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_20x["POSid"] = blt_t1n19_dsaonly_20x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_dsaonly_40x["POSid"] = blt_t1n19_dsaonly_40x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n19_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_10x, callable_pr)
blt_t1n19_dsaonly_10x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_10x, dsa_only_pr_1kb)
blt_t1n19_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_20x, callable_pr)
blt_t1n19_dsaonly_20x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_20x, dsa_only_pr_1kb)
blt_t1n19_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_40x, callable_pr)
blt_t1n19_dsaonly_40x = vcf_in_pyranges_interval(blt_t1n19_dsaonly_40x, dsa_only_pr_1kb)

blt_t1n19_dsaonly_10x = blt_t1n19_dsaonly_10x[~blt_t1n19_dsaonly_10x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_20x = blt_t1n19_dsaonly_20x[~blt_t1n19_dsaonly_20x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n19_dsaonly_40x = blt_t1n19_dsaonly_40x[~blt_t1n19_dsaonly_40x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)

# %%
# INFO: T1N4 100X
blt_t1n4_100x_dsa_only_snv = set(blt_t1n4_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_100x_dsa_only_snv_tp, blt_t1n4_100x_dsa_only_snv_fp, blt_t1n4_100x_dsa_only_snv_fn, blt_t1n4_100x_dsa_only_snv_precision, blt_t1n4_100x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n4_100x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N4 40X
blt_t1n4_40x_dsa_only_snv = set(blt_t1n4_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_40x_dsa_only_snv_tp, blt_t1n4_40x_dsa_only_snv_fp, blt_t1n4_40x_dsa_only_snv_fn, blt_t1n4_40x_dsa_only_snv_precision, blt_t1n4_40x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n4_40x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N4 20X
blt_t1n4_20x_dsa_only_snv = set(blt_t1n4_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n4_20x_dsa_only_snv_tp, blt_t1n4_20x_dsa_only_snv_fp, blt_t1n4_20x_dsa_only_snv_fn, blt_t1n4_20x_dsa_only_snv_precision, blt_t1n4_20x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n4_20x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N4 10X
blt_t1n4_10x_dsa_only_snv = set(blt_t1n4_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n4_10x_dsa_only_snv_tp, blt_t1n4_10x_dsa_only_snv_fp, blt_t1n4_10x_dsa_only_snv_fn, blt_t1n4_10x_dsa_only_snv_precision, blt_t1n4_10x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n4_10x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N9 100X
blt_t1n9_100x_dsa_only_snv = set(blt_t1n9_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_100x_dsa_only_snv_tp, blt_t1n9_100x_dsa_only_snv_fp, blt_t1n9_100x_dsa_only_snv_fn, blt_t1n9_100x_dsa_only_snv_precision, blt_t1n9_100x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n9_100x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N9 40X
blt_t1n9_40x_dsa_only_snv = set(blt_t1n9_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_40x_dsa_only_snv_tp, blt_t1n9_40x_dsa_only_snv_fp, blt_t1n9_40x_dsa_only_snv_fn, blt_t1n9_40x_dsa_only_snv_precision, blt_t1n9_40x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n9_40x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N9 20X
blt_t1n9_20x_dsa_only_snv = set(blt_t1n9_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n9_20x_dsa_only_snv_tp, blt_t1n9_20x_dsa_only_snv_fp, blt_t1n9_20x_dsa_only_snv_fn, blt_t1n9_20x_dsa_only_snv_precision, blt_t1n9_20x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n9_20x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N9 10X
blt_t1n9_10x_dsa_only_snv = set(blt_t1n9_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n9_10x_dsa_only_snv_tp, blt_t1n9_10x_dsa_only_snv_fp, blt_t1n9_10x_dsa_only_snv_fn, blt_t1n9_10x_dsa_only_snv_precision, blt_t1n9_10x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n9_10x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N19 100X
blt_t1n19_100x_dsa_only_snv = set(blt_t1n19_dsaonly_100x["SNVid"].values) - set(bl_100x_snv["SNVid"].values)
blt_t1n19_100x_dsa_only_snv_tp, blt_t1n19_100x_dsa_only_snv_fp, blt_t1n19_100x_dsa_only_snv_fn, blt_t1n19_100x_dsa_only_snv_precision, blt_t1n19_100x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n19_100x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N19 40X
blt_t1n19_40x_dsa_only_snv = set(blt_t1n19_dsaonly_40x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_40x_dsa_only_snv_tp, blt_t1n19_40x_dsa_only_snv_fp, blt_t1n19_40x_dsa_only_snv_fn, blt_t1n19_40x_dsa_only_snv_precision, blt_t1n19_40x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n19_40x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N19 20X
blt_t1n19_20x_dsa_only_snv = set(blt_t1n19_dsaonly_20x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_20x_dsa_only_snv_tp, blt_t1n19_20x_dsa_only_snv_fp, blt_t1n19_20x_dsa_only_snv_fn, blt_t1n19_20x_dsa_only_snv_precision, blt_t1n19_20x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n19_20x_dsa_only_snv, snv_referenceset_dsa_only_set)


# INFO: T1N19 10X
blt_t1n19_10x_dsa_only_snv = set(blt_t1n19_dsaonly_10x["SNVid"].values) - set(bl_100x_snv["SNVid"].values) 
blt_t1n19_10x_dsa_only_snv_tp, blt_t1n19_10x_dsa_only_snv_fp, blt_t1n19_10x_dsa_only_snv_fn, blt_t1n19_10x_dsa_only_snv_precision, blt_t1n19_10x_dsa_only_snv_recall, _ = snv_pr_metrics(blt_t1n19_10x_dsa_only_snv, snv_referenceset_dsa_only_set)



# %%
# ADDED: Illumina short-read (Diploid 100X, haploid 50X) for evaluating mSNV discovery performance
# INFO: Analysis in DSA-space Using short-read DSA-based SNVs that could NOT!! be surjected to T2T-CHM13 (DSA-only SNVs in terms of CHM13)
platform="Illumina"
bl_sr_50x_snv = read_vcf(f"{insilico_dsa_dir}/N_ONLY/{platform}/Variant_Calls/DeepSomatic/COLO829BL_insilico_N_ONLY_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
bl_sr_50x_snv['SNVid'] = bl_sr_50x_snv[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
bl_sr_50x_snv = bl_sr_50x_snv[(vcf_info_getter(bl_sr_50x_snv, "Flagger") == "Hap") & (vcf_info_getter(bl_sr_50x_snv, "NucFlag").isna())].reset_index(drop=True)

bl_sr_50x_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, callable_pr)
bl_sr_50x_snv = vcf_in_pyranges_interval(bl_sr_50x_snv, dsa_only_pr_1kb)

blt_t1n4_sr_dsaonly_50x = read_vcf(f"{insilico_dsa_dir}/T1N4/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N4_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n9_sr_dsaonly_50x = read_vcf(f"{insilico_dsa_dir}/T1N9/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N9_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")
blt_t1n19_sr_dsaonly_50x = read_vcf(f"{insilico_dsa_dir}/T1N19/50X/{platform}/Variant_Calls/DeepSomatic/COLO829BLT_insilico_T1N19_sr_50X.deepsomatictonly.PASS.snv.annot.vcf.gz")

blt_t1n4_sr_dsaonly_50x = blt_t1n4_sr_dsaonly_50x[(vcf_info_getter(blt_t1n4_sr_dsaonly_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n4_sr_dsaonly_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n9_sr_dsaonly_50x = blt_t1n9_sr_dsaonly_50x[(vcf_info_getter(blt_t1n9_sr_dsaonly_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n9_sr_dsaonly_50x, "NucFlag").isna())].reset_index(drop=True)
blt_t1n19_sr_dsaonly_50x = blt_t1n19_sr_dsaonly_50x[(vcf_info_getter(blt_t1n19_sr_dsaonly_50x, "Flagger") == "Hap") & (vcf_info_getter(blt_t1n19_sr_dsaonly_50x, "NucFlag").isna())].reset_index(drop=True)

blt_t1n4_sr_dsaonly_50x['SNVid'] = blt_t1n4_sr_dsaonly_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_dsaonly_50x['SNVid'] = blt_t1n9_sr_dsaonly_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_dsaonly_50x['SNVid'] = blt_t1n19_sr_dsaonly_50x[['CHROM', 'POS', 'REF', 'ALT']].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_dsaonly_50x["POSid"] = blt_t1n4_sr_dsaonly_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n9_sr_dsaonly_50x["POSid"] = blt_t1n9_sr_dsaonly_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)
blt_t1n19_sr_dsaonly_50x["POSid"] = blt_t1n19_sr_dsaonly_50x[["CHROM", "POS"]].astype(str).apply('_'.join, axis=1)

blt_t1n4_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n4_sr_dsaonly_50x, callable_pr)
blt_t1n4_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n4_sr_dsaonly_50x, dsa_only_pr_1kb)
blt_t1n9_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n9_sr_dsaonly_50x, callable_pr)
blt_t1n9_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n9_sr_dsaonly_50x, dsa_only_pr_1kb)
blt_t1n19_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n19_sr_dsaonly_50x, callable_pr)
blt_t1n19_sr_dsaonly_50x = vcf_in_pyranges_interval(blt_t1n19_sr_dsaonly_50x, dsa_only_pr_1kb)

blt_t1n4_sr_dsaonly_50x = blt_t1n4_sr_dsaonly_50x[~blt_t1n4_sr_dsaonly_50x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n9_sr_dsaonly_50x = blt_t1n9_sr_dsaonly_50x[~blt_t1n9_sr_dsaonly_50x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)
blt_t1n19_sr_dsaonly_50x = blt_t1n19_sr_dsaonly_50x[~blt_t1n19_sr_dsaonly_50x["POSid"].isin(snv_flagset_position_set_plus_hg38_surjectable_position)].reset_index(drop=True)

# INFO: T1N4 50X
blt_t1n4_sr_dsaonly_50X_snv = set(blt_t1n4_sr_dsaonly_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n4_sr_dsaonly_50X_snv_tp, blt_t1n4_sr_dsaonly_50X_snv_fp, blt_t1n4_sr_dsaonly_50X_snv_fn, blt_t1n4_sr_dsaonly_50X_snv_precision, blt_t1n4_sr_dsaonly_50X_snv_recall, _ = snv_pr_metrics(blt_t1n4_sr_dsaonly_50X_snv, snv_referenceset_dsa_only_set)


# INFO: T1N9 50X
blt_t1n9_sr_dsaonly_50X_snv = set(blt_t1n9_sr_dsaonly_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n9_sr_dsaonly_50X_snv_tp, blt_t1n9_sr_dsaonly_50X_snv_fp, blt_t1n9_sr_dsaonly_50X_snv_fn, blt_t1n9_sr_dsaonly_50X_snv_precision, blt_t1n9_sr_dsaonly_50X_snv_recall, _ = snv_pr_metrics(blt_t1n9_sr_dsaonly_50X_snv, snv_referenceset_dsa_only_set)


# INFO: T1N19 50X
blt_t1n19_sr_dsaonly_50X_snv = set(blt_t1n19_sr_dsaonly_50x["SNVid"].values) - set(bl_sr_50x_snv["SNVid"].values)
blt_t1n19_sr_dsaonly_50X_snv_tp, blt_t1n19_sr_dsaonly_50X_snv_fp, blt_t1n19_sr_dsaonly_50X_snv_fn, blt_t1n19_sr_dsaonly_50X_snv_precision, blt_t1n19_sr_dsaonly_50X_snv_recall, _ = snv_pr_metrics(blt_t1n19_sr_dsaonly_50X_snv, snv_referenceset_dsa_only_set)

# %%
# INFO: Using `snv_referenceset_between_hap_surjected` to calculate Precision on DSA-only SNVs compared to GRCh38
# INFO: T1N4 50X
blt_t1n4_sr_dsaonly_50X_snv_hap_surject_tp, blt_t1n4_sr_dsaonly_50X_snv_hap_surject_fp, blt_t1n4_sr_dsaonly_50X_snv_hap_surject_fn, blt_t1n4_sr_dsaonly_50X_snv_hap_surject_precision, blt_t1n4_sr_dsaonly_50X_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n4_sr_dsaonly_50X_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N9 50X
blt_t1n9_sr_dsaonly_50X_snv_hap_surject_tp, blt_t1n9_sr_dsaonly_50X_snv_hap_surject_fp, blt_t1n9_sr_dsaonly_50X_snv_hap_surject_fn, blt_t1n9_sr_dsaonly_50X_snv_hap_surject_precision, blt_t1n9_sr_dsaonly_50X_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n9_sr_dsaonly_50X_snv, snv_referenceset_between_hap_surjected)


# INFO: T1N19 50X
blt_t1n19_sr_dsaonly_50X_snv_hap_surject_tp, blt_t1n19_sr_dsaonly_50X_snv_hap_surject_fp, blt_t1n19_sr_dsaonly_50X_snv_hap_surject_fn, blt_t1n19_sr_dsaonly_50X_snv_hap_surject_precision, blt_t1n19_sr_dsaonly_50X_snv_hap_surject_recall, _ = snv_pr_metrics(blt_t1n19_sr_dsaonly_50X_snv, snv_referenceset_between_hap_surjected)


# %%
# INFO: DSA-only SNV (compared to T2T-CHM13) Precision & Recall 
pr_dsa_only_chm13_data = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_100x_dsa_only_chm13_snv_precision, blt_t1n4_40x_dsa_only_chm13_snv_precision, blt_t1n4_20x_dsa_only_chm13_snv_precision, blt_t1n4_10x_dsa_only_chm13_snv_precision,
                         blt_t1n9_100x_dsa_only_chm13_snv_precision, blt_t1n9_40x_dsa_only_chm13_snv_precision, blt_t1n9_20x_dsa_only_chm13_snv_precision, blt_t1n9_10x_dsa_only_chm13_snv_precision,
                         blt_t1n19_100x_dsa_only_chm13_snv_precision, blt_t1n19_40x_dsa_only_chm13_snv_precision, blt_t1n19_20x_dsa_only_chm13_snv_precision, blt_t1n19_10x_dsa_only_chm13_snv_precision],
           "Recall": [blt_t1n4_100x_dsa_only_chm13_snv_recall, blt_t1n4_40x_dsa_only_chm13_snv_recall, blt_t1n4_20x_dsa_only_chm13_snv_recall, blt_t1n4_10x_dsa_only_chm13_snv_recall,
                      blt_t1n9_100x_dsa_only_chm13_snv_recall, blt_t1n9_40x_dsa_only_chm13_snv_recall, blt_t1n9_20x_dsa_only_chm13_snv_recall, blt_t1n9_10x_dsa_only_chm13_snv_recall,
                      blt_t1n19_100x_dsa_only_chm13_snv_recall, blt_t1n19_40x_dsa_only_chm13_snv_recall, blt_t1n19_20x_dsa_only_chm13_snv_recall, blt_t1n19_10x_dsa_only_chm13_snv_recall]}

pr_dsa_only_chm13_df = pd.DataFrame(pr_dsa_only_chm13_data)

pr_dsa_only_chm13_df['Coverage_numeric'] = pr_dsa_only_chm13_df['Coverage'].str.replace('X', '').astype(int)

ordered_ratios = ['T1N4', 'T1N9', 'T1N19']
category_colors = sns.color_palette("Paired", 9)
pr_dsa_only_chm13_df['Ratio'] = pd.Categorical(pr_dsa_only_chm13_df['Ratio'], categories=ordered_ratios, ordered=True)

pr_dsa_only_chm13_df = pr_dsa_only_chm13_df.sort_values(by=['Ratio', 'Coverage_numeric'], ascending=[True, False])

pr_dsa_only_chm13_long = pd.melt(pr_dsa_only_chm13_df, 
                  id_vars=['Ratio', 'Coverage', 'Coverage_numeric'], 
                  value_vars=['Precision', 'Recall'],
                  var_name='Metric', 
                  value_name='Value')

ratio_mapping = {
    'T1N4': '1:4 Tumor:Normal mixture\n(~20% VAF)',
    'T1N9': '1:9 Tumor:Normal mixture\n(~10% VAF)', 
    'T1N19': '1:19 Tumor:Normal mixture\n(~5% VAF)'
}

pr_dsa_only_chm13_long['Ratio_Label'] = pr_dsa_only_chm13_long['Ratio'].map(ratio_mapping)

coverage_order = ['100X', '40X', '20X', '10X']
pr_dsa_only_chm13_long['Coverage'] = pd.Categorical(pr_dsa_only_chm13_long['Coverage'], categories=coverage_order, ordered=True)

plot = (ggplot(pr_dsa_only_chm13_long, aes(x='Coverage', y='Value', group='Ratio')) +
        geom_line(size=1, color='darkred') +
        geom_point(size=2, color='darkred') +
        facet_grid('Metric ~ Ratio_Label', scales='free_y') +  # Use Ratio_Label instead of Ratio
        scale_y_continuous(breaks=np.arange(0, 1.25, 0.25), 
                          labels=['0.0', '0.25', '0.5', '0.75', '1.0'],
                          limits=(0, 1)) +
        labs(title='Precision and Recall by Coverage and Tumor:Normal Ratio',
             x='Sequencing Coverage',
             y='') +
        theme_minimal() +
        theme(
            text=element_text(family='Arial'),
            axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
            axis_text_y=element_text(color='black'),
            axis_title_x=element_text(color='black'),
            axis_title_y=element_text(color='black'),
            plot_title=element_text(color='black'),
            strip_text=element_text(size=10, face='bold', color='black')))
ggsavefig_and_show(plot, "Precision_and_Recall_by_Coverage_and_Ratio_DSA-only_dsa_only_chm13_snvs")

# %%
# INFO: DSA-only SNV (compared to GRCh38) Precision & Recall 
pr_dsa_only_data = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_100x_dsa_only_snv_precision, blt_t1n4_40x_dsa_only_snv_precision, blt_t1n4_20x_dsa_only_snv_precision, blt_t1n4_10x_dsa_only_snv_precision,
                         blt_t1n9_100x_dsa_only_snv_precision, blt_t1n9_40x_dsa_only_snv_precision, blt_t1n9_20x_dsa_only_snv_precision, blt_t1n9_10x_dsa_only_snv_precision,
                         blt_t1n19_100x_dsa_only_snv_precision, blt_t1n19_40x_dsa_only_snv_precision, blt_t1n19_20x_dsa_only_snv_precision, blt_t1n19_10x_dsa_only_snv_precision],
           "Recall": [blt_t1n4_100x_dsa_only_snv_recall, blt_t1n4_40x_dsa_only_snv_recall, blt_t1n4_20x_dsa_only_snv_recall, blt_t1n4_10x_dsa_only_snv_recall,
                      blt_t1n9_100x_dsa_only_snv_recall, blt_t1n9_40x_dsa_only_snv_recall, blt_t1n9_20x_dsa_only_snv_recall, blt_t1n9_10x_dsa_only_snv_recall,
                      blt_t1n19_100x_dsa_only_snv_recall, blt_t1n19_40x_dsa_only_snv_recall, blt_t1n19_20x_dsa_only_snv_recall, blt_t1n19_10x_dsa_only_snv_recall]}

pr_dsa_only_df = pd.DataFrame(pr_dsa_only_data)

pr_dsa_only_df['Coverage_numeric'] = pr_dsa_only_df['Coverage'].str.replace('X', '').astype(int)

ordered_ratios = ['T1N4', 'T1N9', 'T1N19']
category_colors = sns.color_palette("Paired", 9)
pr_dsa_only_df['Ratio'] = pd.Categorical(pr_dsa_only_df['Ratio'], categories=ordered_ratios, ordered=True)

pr_dsa_only_df = pr_dsa_only_df.sort_values(by=['Ratio', 'Coverage_numeric'], ascending=[True, False])

pr_dsa_only_long = pd.melt(pr_dsa_only_df, 
                  id_vars=['Ratio', 'Coverage', 'Coverage_numeric'], 
                  value_vars=['Precision', 'Recall'],
                  var_name='Metric', 
                  value_name='Value')

ratio_mapping = {
    'T1N4': '1:4 Tumor:Normal mixture\n(~20% VAF)',
    'T1N9': '1:9 Tumor:Normal mixture\n(~10% VAF)', 
    'T1N19': '1:19 Tumor:Normal mixture\n(~5% VAF)'
}

pr_dsa_only_long['Ratio_Label'] = pr_dsa_only_long['Ratio'].map(ratio_mapping)

coverage_order = ['100X', '40X', '20X', '10X']
pr_dsa_only_long['Coverage'] = pd.Categorical(pr_dsa_only_long['Coverage'], categories=coverage_order, ordered=True)

plot = (ggplot(pr_dsa_only_long, aes(x='Coverage', y='Value', group='Ratio')) +
        geom_line(size=1, color='darkred') +
        geom_point(size=2, color='darkred') +
        facet_grid('Metric ~ Ratio_Label', scales='free_y') +  # Use Ratio_Label instead of Ratio
        scale_y_continuous(breaks=np.arange(0, 1.25, 0.25), 
                          labels=['0.0', '0.25', '0.5', '0.75', '1.0'],
                          limits=(0, 1)) +
        labs(title='Precision and Recall by Coverage and Tumor:Normal Ratio',
             x='Sequencing Coverage',
             y='') +
        theme_minimal() +
        theme(
              text=element_text(family='Arial'),  
              axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
              axis_text_y=element_text(color='black'),
              axis_title_x=element_text(color='black'),
              axis_title_y=element_text(color='black'),
              plot_title=element_text(color='black'),
              strip_text=element_text(size=10, face='bold', color='black')))
ggsavefig_and_show(plot, "Precision_and_Recall_by_Coverage_and_Ratio_DSA-only_dsa_only_snvs")


############################################################################################################################
############################################################################################################################
############################################################################################################################
# INFO: Plotting Precision-Recall Curve for DeepSomatic Tumor-only-mode Somatic SNVs from in-silico mixtures ###############
############################################################################################################################
############################################################################################################################
############################################################################################################################


# %%
# INFO: With PON Filtering (GRCh38)
pr_data_with_pon_hg38 = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_hg38_pon_100x_snv_precision, blt_t1n4_hg38_pon_40x_snv_precision, blt_t1n4_hg38_pon_20x_snv_precision, blt_t1n4_hg38_pon_10x_snv_precision,
                             blt_t1n9_hg38_pon_100x_snv_precision, blt_t1n9_hg38_pon_40x_snv_precision, blt_t1n9_hg38_pon_20x_snv_precision, blt_t1n9_hg38_pon_10x_snv_precision,
                             blt_t1n19_hg38_pon_100x_snv_precision, blt_t1n19_hg38_pon_40x_snv_precision, blt_t1n19_hg38_pon_20x_snv_precision, blt_t1n19_hg38_pon_10x_snv_precision],
           "Recall": [blt_t1n4_hg38_pon_100x_snv_recall_withreject, blt_t1n4_hg38_pon_40x_snv_recall_withreject, blt_t1n4_hg38_pon_20x_snv_recall_withreject, blt_t1n4_hg38_pon_10x_snv_recall_withreject,
                                  blt_t1n9_hg38_pon_100x_snv_recall_withreject, blt_t1n9_hg38_pon_40x_snv_recall_withreject, blt_t1n9_hg38_pon_20x_snv_recall_withreject, blt_t1n9_hg38_pon_10x_snv_recall_withreject,
                                  blt_t1n19_hg38_pon_100x_snv_recall_withreject, blt_t1n19_hg38_pon_40x_snv_recall_withreject, blt_t1n19_hg38_pon_20x_snv_recall_withreject, blt_t1n19_hg38_pon_10x_snv_recall_withreject]
                        }

pr_df_with_pon_hg38 = pd.DataFrame(pr_data_with_pon_hg38)

# INFO: Without PON Filtering (GRCh38)
pr_data_wo_pon_hg38 = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_hg38_nonpon_100x_snv_precision, blt_t1n4_hg38_nonpon_40x_snv_precision, blt_t1n4_hg38_nonpon_20x_snv_precision, blt_t1n4_hg38_nonpon_10x_snv_precision,
                             blt_t1n9_hg38_nonpon_100x_snv_precision, blt_t1n9_hg38_nonpon_40x_snv_precision, blt_t1n9_hg38_nonpon_20x_snv_precision, blt_t1n9_hg38_nonpon_10x_snv_precision,
                             blt_t1n19_hg38_nonpon_100x_snv_precision, blt_t1n19_hg38_nonpon_40x_snv_precision, blt_t1n19_hg38_nonpon_20x_snv_precision, blt_t1n19_hg38_nonpon_10x_snv_precision],
           "Recall": [blt_t1n4_hg38_nonpon_100x_snv_recall_withreject, blt_t1n4_hg38_nonpon_40x_snv_recall_withreject, blt_t1n4_hg38_nonpon_20x_snv_recall_withreject, blt_t1n4_hg38_nonpon_10x_snv_recall_withreject,
                                  blt_t1n9_hg38_nonpon_100x_snv_recall_withreject, blt_t1n9_hg38_nonpon_40x_snv_recall_withreject, blt_t1n9_hg38_nonpon_20x_snv_recall_withreject, blt_t1n9_hg38_nonpon_10x_snv_recall_withreject,
                                  blt_t1n19_hg38_nonpon_100x_snv_recall_withreject, blt_t1n19_hg38_nonpon_40x_snv_recall_withreject, blt_t1n19_hg38_nonpon_20x_snv_recall_withreject, blt_t1n19_hg38_nonpon_10x_snv_recall_withreject]
                        }

pr_df_wo_pon_hg38 = pd.DataFrame(pr_data_wo_pon_hg38)

# INFO: With PON Filtering (T2T-CHM13)
pr_data_with_pon_chm13 = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_chm13_pon_100x_snv_precision, blt_t1n4_chm13_pon_40x_snv_precision, blt_t1n4_chm13_pon_20x_snv_precision, blt_t1n4_chm13_pon_10x_snv_precision,
                             blt_t1n9_chm13_pon_100x_snv_precision, blt_t1n9_chm13_pon_40x_snv_precision, blt_t1n9_chm13_pon_20x_snv_precision, blt_t1n9_chm13_pon_10x_snv_precision,
                             blt_t1n19_chm13_pon_100x_snv_precision, blt_t1n19_chm13_pon_40x_snv_precision, blt_t1n19_chm13_pon_20x_snv_precision, blt_t1n19_chm13_pon_10x_snv_precision],
           "Recall": [blt_t1n4_chm13_pon_100x_snv_recall_withreject, blt_t1n4_chm13_pon_40x_snv_recall_withreject, blt_t1n4_chm13_pon_20x_snv_recall_withreject, blt_t1n4_chm13_pon_10x_snv_recall_withreject,
                                  blt_t1n9_chm13_pon_100x_snv_recall_withreject, blt_t1n9_chm13_pon_40x_snv_recall_withreject, blt_t1n9_chm13_pon_20x_snv_recall_withreject, blt_t1n9_chm13_pon_10x_snv_recall_withreject,
                                  blt_t1n19_chm13_pon_100x_snv_recall_withreject, blt_t1n19_chm13_pon_40x_snv_recall_withreject, blt_t1n19_chm13_pon_20x_snv_recall_withreject, blt_t1n19_chm13_pon_10x_snv_recall_withreject]
            }

pr_df_with_pon_chm13 = pd.DataFrame(pr_data_with_pon_chm13)

# INFO: Without PON Filtering (T2T-CHM13)
pr_data_wo_pon_chm13 = {"Ratio": ["T1N4", "T1N4", "T1N4", "T1N4", "T1N9", "T1N9", "T1N9", "T1N9", "T1N19", "T1N19", "T1N19", "T1N19"],
           "Coverage": ["100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X", "100X", "40X", "20X", "10X"],
           "Precision": [blt_t1n4_chm13_nonpon_100x_snv_precision, blt_t1n4_chm13_nonpon_40x_snv_precision, blt_t1n4_chm13_nonpon_20x_snv_precision, blt_t1n4_chm13_nonpon_10x_snv_precision,
                             blt_t1n9_chm13_nonpon_100x_snv_precision, blt_t1n9_chm13_nonpon_40x_snv_precision, blt_t1n9_chm13_nonpon_20x_snv_precision, blt_t1n9_chm13_nonpon_10x_snv_precision,
                             blt_t1n19_chm13_nonpon_100x_snv_precision, blt_t1n19_chm13_nonpon_40x_snv_precision, blt_t1n19_chm13_nonpon_20x_snv_precision, blt_t1n19_chm13_nonpon_10x_snv_precision],
           "Recall": [blt_t1n4_chm13_nonpon_100x_snv_recall_withreject, blt_t1n4_chm13_nonpon_40x_snv_recall_withreject, blt_t1n4_chm13_nonpon_20x_snv_recall_withreject, blt_t1n4_chm13_nonpon_10x_snv_recall_withreject,
                                  blt_t1n9_chm13_nonpon_100x_snv_recall_withreject, blt_t1n9_chm13_nonpon_40x_snv_recall_withreject, blt_t1n9_chm13_nonpon_20x_snv_recall_withreject, blt_t1n9_chm13_nonpon_10x_snv_recall_withreject,
                                  blt_t1n19_chm13_nonpon_100x_snv_recall_withreject, blt_t1n19_chm13_nonpon_40x_snv_recall_withreject, blt_t1n19_chm13_nonpon_20x_snv_recall_withreject, blt_t1n19_chm13_nonpon_10x_snv_recall_withreject]
            }

pr_df_wo_pon_chm13 = pd.DataFrame(pr_data_wo_pon_chm13)

# %%
pr_df_labeled = pr_df.copy()
pr_df_labeled['Method'] = 'DSA-based'

pr_df_with_pon_hg38_labeled = pr_df_with_pon_hg38.copy()
pr_df_with_pon_hg38_labeled['Method'] = 'hg38-based-PON'

pr_df_wo_pon_hg38_labeled = pr_df_wo_pon_hg38.copy()
pr_df_wo_pon_hg38_labeled['Method'] = 'hg38-based-non-PON'

pr_df_with_pon_chm13_labeled = pr_df_with_pon_chm13.copy()
pr_df_with_pon_chm13_labeled['Method'] = 'chm13-based-PON'

pr_df_wo_pon_chm13_labeled = pr_df_wo_pon_chm13.copy()
pr_df_wo_pon_chm13_labeled['Method'] = 'chm13-based-non-PON'

pr_dsa_only_df_labeled = pr_dsa_only_df.copy()
pr_dsa_only_df_labeled['Method'] = 'DSA-based-dsa-only-hg38-SNVs'

pr_dsa_only_chm13_df_labeled = pr_dsa_only_chm13_df.copy()
pr_dsa_only_chm13_df_labeled['Method'] = 'DSA-based-dsa-only-chm13-SNVs'

if 'Coverage_numeric' not in pr_df_with_pon_hg38_labeled.columns:
    coverage_to_numeric = {'100X': 100, '40X': 40, '20X': 20, '10X': 10}
    pr_df_with_pon_hg38_labeled['Coverage_numeric'] = pr_df_with_pon_hg38_labeled['Coverage'].map(coverage_to_numeric)
    pr_df_wo_pon_hg38_labeled['Coverage_numeric'] = pr_df_wo_pon_hg38_labeled['Coverage'].map(coverage_to_numeric)
    pr_df_with_pon_chm13_labeled['Coverage_numeric'] = pr_df_with_pon_chm13_labeled['Coverage'].map(coverage_to_numeric)
    pr_df_wo_pon_chm13_labeled['Coverage_numeric'] = pr_df_wo_pon_chm13_labeled['Coverage'].map(coverage_to_numeric)
    pr_dsa_only_df_labeled['Coverage_numeric'] = pr_dsa_only_df_labeled['Coverage'].map(coverage_to_numeric)
    pr_dsa_only_chm13_df_labeled['Coverage_numeric'] = pr_dsa_only_chm13_df_labeled['Coverage'].map(coverage_to_numeric)

combined_df = pd.concat([
    pr_df_labeled, 
    pr_df_with_pon_hg38_labeled, 
    pr_df_wo_pon_hg38_labeled,
    pr_df_with_pon_chm13_labeled, 
    pr_df_wo_pon_chm13_labeled,
    pr_dsa_only_df_labeled, 
    pr_dsa_only_chm13_df_labeled], ignore_index=True)

pr_long = pd.melt(combined_df, 
                  id_vars=['Ratio', 'Coverage', 'Coverage_numeric', 'Method'], 
                  value_vars=['Precision', 'Recall'],
                  var_name='Metric', 
                  value_name='Value')

ratio_mapping = {
    'T1N4': '1:4 Tumor:Normal mixture\n(~20% VAF)',
    'T1N9': '1:9 Tumor:Normal mixture\n(~10% VAF)', 
    'T1N19': '1:19 Tumor:Normal mixture\n(~5% VAF)'
    }

pr_long['Ratio_Label'] = pr_long['Ratio'].map(ratio_mapping)

coverage_order = ['100X', '40X', '20X', '10X']
pr_long['Coverage'] = pd.Categorical(pr_long['Coverage'], categories=coverage_order, ordered=True)

ratio_label_order = [
    '1:4 Tumor:Normal mixture\n(~20% VAF)',
    '1:9 Tumor:Normal mixture\n(~10% VAF)', 
    '1:19 Tumor:Normal mixture\n(~5% VAF)'
]
pr_long['Ratio_Label'] = pd.Categorical(pr_long['Ratio_Label'], categories=ratio_label_order, ordered=True)
pr_long = pr_long.sort_values(by=["Method", "Metric"], ascending=[False, True]).reset_index(drop=True)

dashed_methods = ['hg38-based-non-PON', 'chm13-based-non-PON']
solid_methods = [m for m in pr_long['Method'].unique() if m not in dashed_methods]

plot = (ggplot(pr_long, aes(x='Coverage', y='Value', group='Method', color='Method')) +
        geom_path(data=pr_long[pr_long['Method'].isin(solid_methods)], size=1, linetype='solid') +
        geom_path(data=pr_long[pr_long['Method'].isin(dashed_methods)], size=1, linetype='dashed') +
        geom_point(size=2) +
        facet_grid('Metric ~ Ratio_Label', scales='free_y') +
        scale_color_manual(values={
        'hg38-based-PON': 'grey', 
        'hg38-based-non-PON': 'grey', 
        'chm13-based-PON': 'lightslategray',
        'chm13-based-non-PON': 'darkblue',
        'DSA-based': 'darkred', 
        'DSA-based-dsa-only-hg38-SNVs': 'darkviolet', 
        'DSA-based-dsa-only-chm13-SNVs': 'orchid'},
        guide=guide_legend(nrow=2)) +
        scale_y_continuous(breaks=np.arange(0, 1.25, 0.25), 
                          labels=['0.0', '0.25', '0.5', '0.75', '1.0'],
                          limits=(0, 1)) +
        labs(title='',
             x='Sequencing Coverage',
             y='',
             color='') +
        theme_minimal() +
        theme(text=element_text(family='Arial'),
              axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
              axis_text_y=element_text(color='black'),
              axis_title_x=element_text(color='black'),
              axis_title_y=element_text(color='black'),
              plot_title=element_text(color='black'),
              strip_text=element_text(size=10, face='bold', color='black'),
              legend_text=element_text(size=7),
              legend_position='bottom'))

ggsavefig_and_show(plot, "Precision_and_Recall_by_Coverage_and_Ratio_Combined_pon")
# %%
# INFO: Final Figure 🤞
pr_20x = pr_long[pr_long['Coverage'] == '20X'].copy()

ratio_order = ['T1N19', 'T1N9', 'T1N4']
pr_20x['Ratio'] = pd.Categorical(pr_20x['Ratio'], categories=ratio_order, ordered=True)

unified_data = pr_20x.copy()

unified_data['Panel_Row'] = unified_data['Metric']
unified_data['Panel_Col'] = unified_data['Method'].map({
    'DSA-based': 'Genome-wide', 
    'hg38-based-PON': 'Genome-wide',
    'hg38-based-non-PON': 'Genome-wide',
    'chm13-based-PON': 'Genome-wide',
    'chm13-based-non-PON': 'Genome-wide',
    'DSA-based-dsa-only-hg38-SNVs': 'DSA-only segments',
    'DSA-based-dsa-only-chm13-SNVs': 'DSA-only segments'
})

method_order = ['Genome-wide', 'DSA-only segments']
unified_data['Panel_Col'] = pd.Categorical(unified_data['Panel_Col'], categories=method_order, ordered=True)

color_map = {
        'hg38-based-PON': 'orangered', 
        'hg38-based-non-PON': 'orangered', 
        'chm13-based-PON': 'olive',
        'chm13-based-non-PON': 'olive',
        'DSA-based': 'darkblue', 
        'DSA-based-dsa-only-hg38-SNVs': 'darkviolet', 
        'DSA-based-dsa-only-chm13-SNVs': 'orchid'
}

dashed_methods = ['hg38-based-non-PON', 'chm13-based-non-PON']
solid_methods = [m for m in unified_data['Method'].unique() if m not in dashed_methods]

shape_map = {
    'hg38-based-PON': 'o',
    'hg38-based-non-PON': 'o',
    'chm13-based-PON': 'o',
    'chm13-based-non-PON': 'o',
    'DSA-based': 'o',
    'DSA-based-dsa-only-hg38-SNVs': 'o',
    'DSA-based-dsa-only-chm13-SNVs': 'o'
}

unified_data['Shape'] = unified_data['Method'].map(shape_map)

unified_plot = (ggplot(unified_data, aes(x='Ratio', y='Value', group='Method', color='Method', shape='Method')) +
                geom_line(data=unified_data[unified_data['Method'].isin(solid_methods)], size=0.5, linetype='solid') +
                geom_line(data=unified_data[unified_data['Method'].isin(dashed_methods)], size=0.5, linetype='dashed') +
                geom_point(size=2) +
                facet_grid('Panel_Row ~ Panel_Col', scales='free') +
                scale_color_manual(values=color_map, guide=guide_legend(nrow=3)) +
                scale_shape_manual(values={
                    'hg38-based-PON': 'o',
                    'hg38-based-non-PON': 'o',
                    'chm13-based-PON': 'o',
                    'chm13-based-non-PON': 'o',
                    'DSA-based': 'o',
                    'DSA-based-dsa-only-hg38-SNVs': 'o',
                    'DSA-based-dsa-only-chm13-SNVs': 'o'},
                    guide=guide_legend(nrow=3)) +
                scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                labs(x='Tumor:Normal Ratio', y='', color='', shape='') +
                theme_minimal() +
                theme(
                    text=element_text(family='Arial'),
                    axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                    axis_text_y=element_text(color='black'),
                    axis_title_x=element_text(color='black'),
                    axis_title_y=element_text(color='black'),
                    plot_title=element_text(color='black'),
                    strip_text=element_text(size=12, face='bold', color='black'),
                    legend_position='bottom'))

ggsavefig_and_show(unified_plot, "Precision_Recall_20X_Genome-wide_and_DSA-only-segments", width=8, height=6.5)

# %%
# ADDED: Same figure as above, plus short-read (Illumina) points.
# NOTE: short-read runs are 100X total (50X haploid), not 20X like the rest of the figure.
#       No DSA-only-segment short-read metrics yet -> Genome-wide panel only.
sr_rows = pd.DataFrame(
    [('T1N4',  'DSA-based-short-read',   'Precision', blt_t1n4_sr_50x_snv_precision),
     ('T1N4',  'DSA-based-short-read',   'Recall',    blt_t1n4_sr_50x_snv_recall),
     ('T1N9',  'DSA-based-short-read',   'Precision', blt_t1n9_sr_50x_snv_precision),
     ('T1N9',  'DSA-based-short-read',   'Recall',    blt_t1n9_sr_50x_snv_recall),
     ('T1N19', 'DSA-based-short-read',   'Precision', blt_t1n19_sr_50x_snv_precision),
     ('T1N19', 'DSA-based-short-read',   'Recall',    blt_t1n19_sr_50x_snv_recall),
     ('T1N4',  'hg38-based-short-read',  'Precision', blt_t1n4_hg38_sr_50x_snv_precision),
     ('T1N4',  'hg38-based-short-read',  'Recall',    blt_t1n4_hg38_sr_50x_snv_recall_withreject),
     ('T1N9',  'hg38-based-short-read',  'Precision', blt_t1n9_hg38_sr_50x_snv_precision),
     ('T1N9',  'hg38-based-short-read',  'Recall',    blt_t1n9_hg38_sr_50x_snv_recall_withreject),
     ('T1N19', 'hg38-based-short-read',  'Precision', blt_t1n19_hg38_sr_50x_snv_precision),
     ('T1N19', 'hg38-based-short-read',  'Recall',    blt_t1n19_hg38_sr_50x_snv_recall_withreject),
     ('T1N4',  'chm13-based-short-read', 'Precision', blt_t1n4_chm13_sr_50x_snv_precision),
     ('T1N4',  'chm13-based-short-read', 'Recall',    blt_t1n4_chm13_sr_50x_snv_recall_withreject),
     ('T1N9',  'chm13-based-short-read', 'Precision', blt_t1n9_chm13_sr_50x_snv_precision),
     ('T1N9',  'chm13-based-short-read', 'Recall',    blt_t1n9_chm13_sr_50x_snv_recall_withreject),
     ('T1N19', 'chm13-based-short-read', 'Precision', blt_t1n19_chm13_sr_50x_snv_precision),
     ('T1N19', 'chm13-based-short-read', 'Recall',    blt_t1n19_chm13_sr_50x_snv_recall_withreject),
     # DSA-only segments: no "chm13" in the name -> scored against snv_referenceset_dsa_only_set (hg38)
     ('T1N4',  'DSA-based-dsa-only-hg38-SNVs-short-read',  'Precision', blt_t1n4_sr_dsaonly_50X_snv_precision),
     ('T1N4',  'DSA-based-dsa-only-hg38-SNVs-short-read',  'Recall',    blt_t1n4_sr_dsaonly_50X_snv_recall),
     ('T1N9',  'DSA-based-dsa-only-hg38-SNVs-short-read',  'Precision', blt_t1n9_sr_dsaonly_50X_snv_precision),
     ('T1N9',  'DSA-based-dsa-only-hg38-SNVs-short-read',  'Recall',    blt_t1n9_sr_dsaonly_50X_snv_recall),
     ('T1N19', 'DSA-based-dsa-only-hg38-SNVs-short-read',  'Precision', blt_t1n19_sr_dsaonly_50X_snv_precision),
     ('T1N19', 'DSA-based-dsa-only-hg38-SNVs-short-read',  'Recall',    blt_t1n19_sr_dsaonly_50X_snv_recall),
     # "chm13" in the name -> scored against snv_referenceset_dsa_only_chm13_set
     ('T1N4',  'DSA-based-dsa-only-chm13-SNVs-short-read', 'Precision', blt_t1n4_sr_dsaonly_50X_chm13_snv_precision),
     ('T1N4',  'DSA-based-dsa-only-chm13-SNVs-short-read', 'Recall',    blt_t1n4_sr_dsaonly_50X_chm13_snv_recall),
     ('T1N9',  'DSA-based-dsa-only-chm13-SNVs-short-read', 'Precision', blt_t1n9_sr_dsaonly_50X_chm13_snv_precision),
     ('T1N9',  'DSA-based-dsa-only-chm13-SNVs-short-read', 'Recall',    blt_t1n9_sr_dsaonly_50X_chm13_snv_recall),
     ('T1N19', 'DSA-based-dsa-only-chm13-SNVs-short-read', 'Precision', blt_t1n19_sr_dsaonly_50X_chm13_snv_precision),
     ('T1N19', 'DSA-based-dsa-only-chm13-SNVs-short-read', 'Recall',    blt_t1n19_sr_dsaonly_50X_chm13_snv_recall)],
    columns=['Ratio', 'Method', 'Metric', 'Value'])
sr_rows['Panel_Row'] = sr_rows['Metric']
sr_rows['Panel_Col'] = np.where(sr_rows['Method'].str.contains('dsa-only'), 'DSA-only segments', 'Genome-wide')

sr_methods = ['DSA-based-short-read', 'hg38-based-short-read', 'chm13-based-short-read',
              'DSA-based-dsa-only-hg38-SNVs-short-read', 'DSA-based-dsa-only-chm13-SNVs-short-read']

unified_data_sr = pd.concat([unified_data, sr_rows], ignore_index=True)
# NOTE: concat drops the Categorical dtype, so re-apply the orderings
unified_data_sr['Ratio'] = pd.Categorical(unified_data_sr['Ratio'], categories=ratio_order, ordered=True)
unified_data_sr['Panel_Col'] = pd.Categorical(unified_data_sr['Panel_Col'], categories=method_order, ordered=True)

assert len(unified_data_sr) == len(unified_data) + 30
assert unified_data_sr[['Ratio', 'Value', 'Method', 'Panel_Row', 'Panel_Col']].notna().all().all()
assert unified_data_sr['Value'].between(0, 1).all()

color_map_sr = dict(color_map,
                    **{'DSA-based-short-read': 'darkblue',
                       'hg38-based-short-read': 'orangered',
                       'chm13-based-short-read': 'olive',
                       'DSA-based-dsa-only-hg38-SNVs-short-read': 'darkviolet',
                       'DSA-based-dsa-only-chm13-SNVs-short-read': 'orchid'})
shape_map_sr = {m: ('^' if m in sr_methods else 'o') for m in color_map_sr}

dashed_methods = ['hg38-based-non-PON', 'chm13-based-non-PON']
# NOTE: short-read shares its colour with the matching long-read reference,
#       so it gets a dotted line + triangle marker to stay distinguishable
dotted_methods = sr_methods
solid_methods = [m for m in unified_data_sr['Method'].unique()
                 if m not in dashed_methods + dotted_methods]

unified_plot_sr = (ggplot(unified_data_sr, aes(x='Ratio', y='Value', group='Method', color='Method', shape='Method')) +
                   geom_line(data=unified_data_sr[unified_data_sr['Method'].isin(solid_methods)], size=0.5, linetype='solid') +
                   geom_line(data=unified_data_sr[unified_data_sr['Method'].isin(dashed_methods)], size=0.5, linetype='dashed') +
                   geom_line(data=unified_data_sr[unified_data_sr['Method'].isin(dotted_methods)], size=0.5, linetype='dotted') +
                   geom_point(size=2) +
                   facet_grid('Panel_Row ~ Panel_Col', scales='free') +
                   scale_color_manual(values=color_map_sr, guide=guide_legend(nrow=4)) +
                   scale_shape_manual(values=shape_map_sr, guide=guide_legend(nrow=4)) +
                   scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                   scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                   labs(x='Tumor:Normal Ratio', y='', color='', shape='') +
                   theme_minimal() +
                   theme(
                       text=element_text(family='Arial'),
                       axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                       axis_text_y=element_text(color='black'),
                       axis_title_x=element_text(color='black'),
                       axis_title_y=element_text(color='black'),
                       plot_title=element_text(color='black'),
                       strip_text=element_text(size=12, face='bold', color='black'),
                       legend_position='bottom'))

ggsavefig_and_show(unified_plot_sr, "Precision_Recall_20X_Genome-wide_and_DSA-only-segments_with_short-read", width=8, height=7)

# %%
# ADDED: Same as above, but PON-filtered only (drops the non-PON hg38/chm13 series).
# NOTE: with non-PON gone there is no dashed group left, only solid + dotted (short-read).
unified_data_sr_pon = unified_data_sr[~unified_data_sr['Method'].isin(dashed_methods)].copy()

assert not unified_data_sr_pon['Method'].str.contains('non-PON').any()
assert set(unified_data_sr_pon['Method']) == set(unified_data_sr['Method']) - set(dashed_methods)

solid_methods_pon = [m for m in unified_data_sr_pon['Method'].unique() if m not in dotted_methods]

unified_plot_sr_pon = (ggplot(unified_data_sr_pon, aes(x='Ratio', y='Value', group='Method', color='Method', shape='Method')) +
                       geom_line(data=unified_data_sr_pon[unified_data_sr_pon['Method'].isin(solid_methods_pon)], size=0.5, linetype='solid') +
                       geom_line(data=unified_data_sr_pon[unified_data_sr_pon['Method'].isin(dotted_methods)], size=0.5, linetype='dotted') +
                       geom_point(size=2) +
                       facet_grid('Panel_Row ~ Panel_Col', scales='free') +
                       scale_color_manual(values=color_map_sr, guide=guide_legend(nrow=3)) +
                       scale_shape_manual(values=shape_map_sr, guide=guide_legend(nrow=3)) +
                       scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                       scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                       labs(x='Tumor:Normal Ratio', y='', color='', shape='') +
                       theme_minimal() +
                       theme(
                           text=element_text(family='Arial'),
                           axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                           axis_text_y=element_text(color='black'),
                           axis_title_x=element_text(color='black'),
                           axis_title_y=element_text(color='black'),
                           plot_title=element_text(color='black'),
                           strip_text=element_text(size=12, face='bold', color='black'),
                           legend_position='bottom'))

ggsavefig_and_show(unified_plot_sr_pon, "Precision_Recall_20X_Genome-wide_and_DSA-only-segments_with_short-read_PON-only", width=8, height=7)


# %%
# ADDED: Separate plot — short-read PRECISION scored against the between-haplotype-surjected
#        reference (snv_referenceset_between_hap_surjected, L434), for the genome-wide call set
#        (left panel) and the two dsa-only-segment call sets (right panel: hg38 dsa-only, chm13
#        dsa-only).
# NOTE: that reference is a SUPERSET of snv_referenceset, so this is an alternative, more-lenient
#       precision -> precision only, no recall. All three series are scored against the SAME
#       hap-surjected reference; the "hg38"/"chm13" split is in the CALLED set only (unlike the
#       original dsa-only precision, which used hg38- vs chm13-specific dsa-only truth sets).
#       The existing unified_plot_sr / unified_plot_sr_pon are left untouched.
hs_method_gw    = 'DSA-based-hap-surjected-short-read'
hs_method_hg38  = 'DSA-based-dsa-only-hg38-SNVs-hap-surjected-short-read'
hs_method_chm13 = 'DSA-based-dsa-only-chm13-SNVs-hap-surjected-short-read'
hs_rows = pd.DataFrame(
    [('T1N4',  hs_method_gw,    'Precision', blt_t1n4_sr_50x_snv_hap_surject_precision),
     ('T1N9',  hs_method_gw,    'Precision', blt_t1n9_sr_50x_snv_hap_surject_precision),
     ('T1N19', hs_method_gw,    'Precision', blt_t1n19_sr_50x_snv_hap_surject_precision),
     # DSA-only segments (right panel): no "chm13" in var -> hg38 dsa-only calls; "chm13" -> chm13
     ('T1N4',  hs_method_hg38,  'Precision', blt_t1n4_sr_dsaonly_50X_snv_hap_surject_precision),
     ('T1N9',  hs_method_hg38,  'Precision', blt_t1n9_sr_dsaonly_50X_snv_hap_surject_precision),
     ('T1N19', hs_method_hg38,  'Precision', blt_t1n19_sr_dsaonly_50X_snv_hap_surject_precision),
     ('T1N4',  hs_method_chm13, 'Precision', blt_t1n4_sr_dsaonly_50X_chm13_snv_hap_surject_precision),
     ('T1N9',  hs_method_chm13, 'Precision', blt_t1n9_sr_dsaonly_50X_chm13_snv_hap_surject_precision),
     ('T1N19', hs_method_chm13, 'Precision', blt_t1n19_sr_dsaonly_50X_chm13_snv_hap_surject_precision)],
    columns=['Ratio', 'Method', 'Metric', 'Value'])
hs_rows['Panel_Row'] = hs_rows['Metric']
hs_rows['Panel_Col'] = np.where(hs_rows['Method'].str.contains('dsa-only'), 'DSA-only segments', 'Genome-wide')

unified_data_sr_hs = pd.concat([unified_data_sr, hs_rows], ignore_index=True)
# NOTE: concat drops the Categorical dtype, so re-apply the orderings
unified_data_sr_hs['Ratio'] = pd.Categorical(unified_data_sr_hs['Ratio'], categories=ratio_order, ordered=True)
unified_data_sr_hs['Panel_Col'] = pd.Categorical(unified_data_sr_hs['Panel_Col'], categories=method_order, ordered=True)

assert len(unified_data_sr_hs) == len(unified_data_sr) + 9
assert unified_data_sr_hs[['Ratio', 'Value', 'Method', 'Panel_Row', 'Panel_Col']].notna().all().all()
assert unified_data_sr_hs['Value'].between(0, 1).all()

# fresh copies so re-running this cell can't perturb the earlier plots' maps
color_map_sr_hs = dict(color_map_sr,
                       **{hs_method_gw:    'deepskyblue',
                          hs_method_hg38:  'violet',
                          hs_method_chm13: 'deeppink'})
shape_map_sr_hs = dict(shape_map_sr,
                       **{hs_method_gw: '^', hs_method_hg38: '^', hs_method_chm13: '^'})

# hap-surjected series get a dashdot line to set them apart from the dotted short-read originals
dashdot_methods_hs = [hs_method_gw, hs_method_hg38, hs_method_chm13]
solid_methods_hs = [m for m in unified_data_sr_hs['Method'].unique()
                    if m not in dashed_methods + dotted_methods + dashdot_methods_hs]

unified_plot_sr_hs = (ggplot(unified_data_sr_hs, aes(x='Ratio', y='Value', group='Method', color='Method', shape='Method')) +
                      geom_line(data=unified_data_sr_hs[unified_data_sr_hs['Method'].isin(solid_methods_hs)], size=0.5, linetype='solid') +
                      geom_line(data=unified_data_sr_hs[unified_data_sr_hs['Method'].isin(dashed_methods)], size=0.5, linetype='dashed') +
                      geom_line(data=unified_data_sr_hs[unified_data_sr_hs['Method'].isin(dotted_methods)], size=0.5, linetype='dotted') +
                      geom_line(data=unified_data_sr_hs[unified_data_sr_hs['Method'].isin(dashdot_methods_hs)], size=0.5, linetype='dashdot') +
                      geom_point(size=2) +
                      facet_grid('Panel_Row ~ Panel_Col', scales='free') +
                      scale_color_manual(values=color_map_sr_hs, guide=guide_legend(nrow=5)) +
                      scale_shape_manual(values=shape_map_sr_hs, guide=guide_legend(nrow=5)) +
                      scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                      scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                      labs(x='Tumor:Normal Ratio', y='', color='', shape='') +
                      theme_minimal() +
                      theme(
                          text=element_text(family='Arial'),
                          axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                          axis_text_y=element_text(color='black'),
                          axis_title_x=element_text(color='black'),
                          axis_title_y=element_text(color='black'),
                          plot_title=element_text(color='black'),
                          strip_text=element_text(size=12, face='bold', color='black'),
                          legend_position='bottom',
                          legend_text=element_text(size=5))
                          )

ggsavefig_and_show(unified_plot_sr_hs, "Precision_Recall_20X_Genome-wide_and_DSA-only-segments_with_short-read_hap-surjected", width=8, height=7)


# %%
# INFO: Make a bed file of True Positives and False positives of T2T-CHM13-based DeepSomatic Runs
# NOTE: These will be inputs to karyoPloteR

chm13_outdir = "/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/in_silico_mixture/chm13/karyoploteR"

blt_t1n4_chm13_nonpon_20x["Eval"] = blt_t1n4_chm13_nonpon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

blt_t1n9_chm13_nonpon_20x["Eval"] = blt_t1n9_chm13_nonpon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

blt_t1n19_chm13_nonpon_20x["Eval"] = blt_t1n19_chm13_nonpon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

blt_t1n4_chm13_pon_20x["Eval"] = blt_t1n4_chm13_pon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

blt_t1n9_chm13_pon_20x["Eval"] = blt_t1n9_chm13_pon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

blt_t1n19_chm13_pon_20x["Eval"] = blt_t1n19_chm13_pon_20x["SNVid_chm13"].isin(
                                                        snv_referenceset_chm13_primary_set
                                                        ).map({True: "TP", False: "FP"})

print("T1N4 CHM13 non-PON 20X:")
pd.concat([blt_t1n4_chm13_nonpon_20x["CHROM"], blt_t1n4_chm13_nonpon_20x["POS"]-1, blt_t1n4_chm13_nonpon_20x["POS"], blt_t1n4_chm13_nonpon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n4_chm13_nonpon_20x.bed", header=False, index=False, sep="\t")

print("T1N9 CHM13 non-PON 20X:")
pd.concat([blt_t1n9_chm13_nonpon_20x["CHROM"], blt_t1n9_chm13_nonpon_20x["POS"]-1, blt_t1n9_chm13_nonpon_20x["POS"], blt_t1n9_chm13_nonpon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n9_chm13_nonpon_20x.bed", header=False, index=False, sep="\t")

print("T1N19 CHM13 non-PON 20X:")
pd.concat([blt_t1n19_chm13_nonpon_20x["CHROM"], blt_t1n19_chm13_nonpon_20x["POS"]-1, blt_t1n19_chm13_nonpon_20x["POS"], blt_t1n19_chm13_nonpon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n19_chm13_nonpon_20x.bed", header=False, index=False, sep="\t")

print("T1N4 CHM13 PON 20X:")
pd.concat([blt_t1n4_chm13_pon_20x["CHROM"], blt_t1n4_chm13_pon_20x["POS"]-1, blt_t1n4_chm13_pon_20x["POS"], blt_t1n4_chm13_pon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n4_chm13_pon_20x.bed", header=False, index=False, sep="\t")

print("T1N9 CHM13 PON 20X:")
pd.concat([blt_t1n9_chm13_pon_20x["CHROM"], blt_t1n9_chm13_pon_20x["POS"]-1, blt_t1n9_chm13_pon_20x["POS"], blt_t1n9_chm13_pon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n9_chm13_pon_20x.bed", header=False, index=False, sep="\t")

print("T1N19 CHM13 PON 20X:")
pd.concat([blt_t1n19_chm13_pon_20x["CHROM"], blt_t1n19_chm13_pon_20x["POS"]-1, blt_t1n19_chm13_pon_20x["POS"], blt_t1n19_chm13_pon_20x["Eval"]], axis=1).to_csv(f"{chm13_outdir}/blt_t1n19_chm13_pon_20x.bed", header=False, index=False, sep="\t")

# NOTE: How we calculated DSA SNVs surjected to GRCh38 or T2T-CHM13
# awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ {print}' /mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toGRCh38/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_GRCh38_withTags_sorted.bed | cut -f4 | hist | wc -l
#awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/ {print}' /mmfs1/gscratch/stergachislab/mhsohny/SMaHT/Improving_SomaticVariantCalling_through_DSA/ReferenceSet/COLO829BL_DSA_v3.0.0/01.SNV/toCHM13/COLO829T_PassageB_DSA.deepvariant.PASS.snv.annot.shared.scna.dsg.density.filtered_peaks__surj_onto_CHM13_withTags_sorted.bed | hist | wc -l

# %%
# ADDED: Separate plot — Non-Satellite-region DSA precision/recall.
# NOTE: long-read at 20X only (matching the main 20X figure); short-read at sr_50x; plus the
#       short-read hap-surjected series (scored vs snv_referenceset_between_hap_surjected), which
#       is PRECISION only (recall against that padded reference is not meaningful). Standalone —
#       does not touch the earlier plots.
ns_lr   = 'DSA-based (20X)'
ns_sr   = 'DSA-based short-read'
ns_srhs = 'DSA-based short-read (hap-surjected)'
ns_rows = pd.DataFrame(
    [('T1N4',  ns_lr,   'Precision', blt_t1n4_20x_non_satellite_snv_precision),
     ('T1N4',  ns_lr,   'Recall',    blt_t1n4_20x_non_satellite_snv_recall),
     ('T1N9',  ns_lr,   'Precision', blt_t1n9_20x_non_satellite_snv_precision),
     ('T1N9',  ns_lr,   'Recall',    blt_t1n9_20x_non_satellite_snv_recall),
     ('T1N19', ns_lr,   'Precision', blt_t1n19_20x_non_satellite_snv_precision),
     ('T1N19', ns_lr,   'Recall',    blt_t1n19_20x_non_satellite_snv_recall),
     ('T1N4',  ns_sr,   'Precision', blt_t1n4_sr_50x_non_satellite_snv_precision),
     ('T1N4',  ns_sr,   'Recall',    blt_t1n4_sr_50x_non_satellite_snv_recall),
     ('T1N9',  ns_sr,   'Precision', blt_t1n9_sr_50x_non_satellite_snv_precision),
     ('T1N9',  ns_sr,   'Recall',    blt_t1n9_sr_50x_non_satellite_snv_recall),
     ('T1N19', ns_sr,   'Precision', blt_t1n19_sr_50x_non_satellite_snv_precision),
     ('T1N19', ns_sr,   'Recall',    blt_t1n19_sr_50x_non_satellite_snv_recall),
     # hap-surjected: precision only
     ('T1N4',  ns_srhs, 'Precision', blt_t1n4_sr_50x_non_satellite_snv_hap_surject_precision),
     ('T1N9',  ns_srhs, 'Precision', blt_t1n9_sr_50x_non_satellite_snv_hap_surject_precision),
     ('T1N19', ns_srhs, 'Precision', blt_t1n19_sr_50x_non_satellite_snv_hap_surject_precision)],
    columns=['Ratio', 'Method', 'Metric', 'Value'])
ns_rows['Ratio'] = pd.Categorical(ns_rows['Ratio'], categories=ratio_order, ordered=True)

assert len(ns_rows) == 15
assert ns_rows['Value'].between(0, 1).all()

ns_color_map = {ns_lr: 'darkblue', ns_sr: 'darkorange', ns_srhs: 'deeppink'}
ns_shape_map = {ns_lr: 'o',        ns_sr: '^',          ns_srhs: '^'}
# long-read solid, short-read dotted, hap-surjected dashdot (same convention as the SR plots)
ns_solid, ns_dotted, ns_dashdot = [ns_lr], [ns_sr], [ns_srhs]

non_satellite_plot = (ggplot(ns_rows, aes(x='Ratio', y='Value', group='Method', color='Method', shape='Method')) +
                      geom_line(data=ns_rows[ns_rows['Method'].isin(ns_solid)],   size=0.5, linetype='solid') +
                      geom_line(data=ns_rows[ns_rows['Method'].isin(ns_dotted)],  size=0.5, linetype='dotted') +
                      geom_line(data=ns_rows[ns_rows['Method'].isin(ns_dashdot)], size=0.5, linetype='dashdot') +
                      geom_point(size=2) +
                      facet_grid('Metric ~ .') +
                      scale_color_manual(values=ns_color_map) +
                      scale_shape_manual(values=ns_shape_map) +
                      scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                      scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                      labs(title='Non-Satellite regions (DSA)', x='Tumor:Normal Ratio', y='', color='', shape='') +
                      theme_minimal() +
                      theme(
                          text=element_text(family='Arial'),
                          axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                          axis_text_y=element_text(color='black'),
                          axis_title_x=element_text(color='black'),
                          axis_title_y=element_text(color='black'),
                          plot_title=element_text(color='black'),
                          strip_text=element_text(size=12, face='bold', color='black'),
                          legend_position='bottom'))

ggsavefig_and_show(non_satellite_plot, "Precision_Recall_Non-Satellite_20X_and_short-read", width=7, height=6)


# %%
# ADDED: Non-Satellite precision / recall-with-reject — single panel per metric, 6 lines.
# color = reference method (DSA / hg38 / chm13); linetype + shape = modality (PacBio 20X solid/circle,
# short-read 50X dotted/triangle). Recall row uses recall_withreject for hg38 & chm13; DSA has no
# rejection term (it IS the reference) so it uses plain recall, exactly as the main unified_plot does.
# Standalone; the earlier DSA-only non_satellite_plot (which also carries the hap-surjected series)
# is left untouched.
nsf_rows = pd.DataFrame(
    [
     # --- PacBio (40X) ---
     ('T1N4',  'DSA-based',   'PacBio (40X)',     'Precision', blt_t1n4_20x_non_satellite_snv_precision),
     ('T1N4',  'DSA-based',   'PacBio (40X)',     'Recall',    blt_t1n4_20x_non_satellite_snv_recall),
     ('T1N9',  'DSA-based',   'PacBio (40X)',     'Precision', blt_t1n9_20x_non_satellite_snv_precision),
     ('T1N9',  'DSA-based',   'PacBio (40X)',     'Recall',    blt_t1n9_20x_non_satellite_snv_recall),
     ('T1N19', 'DSA-based',   'PacBio (40X)',     'Precision', blt_t1n19_20x_non_satellite_snv_precision),
     ('T1N19', 'DSA-based',   'PacBio (40X)',     'Recall',    blt_t1n19_20x_non_satellite_snv_recall),
     ('T1N4',  'hg38-based',  'PacBio (40X)',     'Precision', blt_t1n4_hg38_pon_20x_non_satellite_snv_precision),
     ('T1N4',  'hg38-based',  'PacBio (40X)',     'Recall',    blt_t1n4_hg38_pon_20x_non_satellite_snv_recall_withreject),
     ('T1N9',  'hg38-based',  'PacBio (40X)',     'Precision', blt_t1n9_hg38_pon_20x_non_satellite_snv_precision),
     ('T1N9',  'hg38-based',  'PacBio (40X)',     'Recall',    blt_t1n9_hg38_pon_20x_non_satellite_snv_recall_withreject),
     ('T1N19', 'hg38-based',  'PacBio (40X)',     'Precision', blt_t1n19_hg38_pon_20x_non_satellite_snv_precision),
     ('T1N19', 'hg38-based',  'PacBio (40X)',     'Recall',    blt_t1n19_hg38_pon_20x_non_satellite_snv_recall_withreject),
     ('T1N4',  'chm13-based', 'PacBio (40X)',     'Precision', blt_t1n4_chm13_pon_20x_non_satellite_snv_precision),
     ('T1N4',  'chm13-based', 'PacBio (40X)',     'Recall',    blt_t1n4_chm13_pon_20x_non_satellite_snv_recall_withreject),
     ('T1N9',  'chm13-based', 'PacBio (40X)',     'Precision', blt_t1n9_chm13_pon_20x_non_satellite_snv_precision),
     ('T1N9',  'chm13-based', 'PacBio (40X)',     'Recall',    blt_t1n9_chm13_pon_20x_non_satellite_snv_recall_withreject),
     ('T1N19', 'chm13-based', 'PacBio (40X)',     'Precision', blt_t1n19_chm13_pon_20x_non_satellite_snv_precision),
     ('T1N19', 'chm13-based', 'PacBio (40X)',     'Recall',    blt_t1n19_chm13_pon_20x_non_satellite_snv_recall_withreject),
     # --- Short-read (100X) ---
     ('T1N4',  'DSA-based',   'Short-read (100X)', 'Precision', blt_t1n4_sr_50x_non_satellite_snv_precision),
     ('T1N4',  'DSA-based',   'Short-read (100X)', 'Recall',    blt_t1n4_sr_50x_non_satellite_snv_recall),
     ('T1N9',  'DSA-based',   'Short-read (100X)', 'Precision', blt_t1n9_sr_50x_non_satellite_snv_precision),
     ('T1N9',  'DSA-based',   'Short-read (100X)', 'Recall',    blt_t1n9_sr_50x_non_satellite_snv_recall),
     ('T1N19', 'DSA-based',   'Short-read (100X)', 'Precision', blt_t1n19_sr_50x_non_satellite_snv_precision),
     ('T1N19', 'DSA-based',   'Short-read (100X)', 'Recall',    blt_t1n19_sr_50x_non_satellite_snv_recall),
     ('T1N4',  'hg38-based',  'Short-read (100X)', 'Precision', blt_t1n4_hg38_sr_50x_non_satellite_snv_precision),
     ('T1N4',  'hg38-based',  'Short-read (100X)', 'Recall',    blt_t1n4_hg38_sr_50x_non_satellite_snv_recall_withreject),
     ('T1N9',  'hg38-based',  'Short-read (100X)', 'Precision', blt_t1n9_hg38_sr_50x_non_satellite_snv_precision),
     ('T1N9',  'hg38-based',  'Short-read (100X)', 'Recall',    blt_t1n9_hg38_sr_50x_non_satellite_snv_recall_withreject),
     ('T1N19', 'hg38-based',  'Short-read (100X)', 'Precision', blt_t1n19_hg38_sr_50x_non_satellite_snv_precision),
     ('T1N19', 'hg38-based',  'Short-read (100X)', 'Recall',    blt_t1n19_hg38_sr_50x_non_satellite_snv_recall_withreject),
     ('T1N4',  'chm13-based', 'Short-read (100X)', 'Precision', blt_t1n4_chm13_sr_50x_non_satellite_snv_precision),
     ('T1N4',  'chm13-based', 'Short-read (100X)', 'Recall',    blt_t1n4_chm13_sr_50x_non_satellite_snv_recall_withreject),
     ('T1N9',  'chm13-based', 'Short-read (100X)', 'Precision', blt_t1n9_chm13_sr_50x_non_satellite_snv_precision),
     ('T1N9',  'chm13-based', 'Short-read (100X)', 'Recall',    blt_t1n9_chm13_sr_50x_non_satellite_snv_recall_withreject),
     ('T1N19', 'chm13-based', 'Short-read (100X)', 'Precision', blt_t1n19_chm13_sr_50x_non_satellite_snv_precision),
     ('T1N19', 'chm13-based', 'Short-read (100X)', 'Recall',    blt_t1n19_chm13_sr_50x_non_satellite_snv_recall_withreject),
     # --- Short-read hap-surjected (100X), precision only ---
     ('T1N4',  'DSA-based', 'Short-read hap-surj (100X)', 'Precision', blt_t1n4_sr_50x_non_satellite_snv_hap_surject_precision),
     ('T1N9',  'DSA-based', 'Short-read hap-surj (100X)', 'Precision', blt_t1n9_sr_50x_non_satellite_snv_hap_surject_precision),
     ('T1N19', 'DSA-based', 'Short-read hap-surj (100X)', 'Precision', blt_t1n19_sr_50x_non_satellite_snv_hap_surject_precision),
    ],
    columns=['Ratio', 'Method', 'Modality', 'Metric', 'Value'])
nsf_rows['Ratio'] = pd.Categorical(nsf_rows['Ratio'], categories=ratio_order, ordered=True)
nsf_rows['Series'] = nsf_rows['Method'] + ' / ' + nsf_rows['Modality']

assert len(nsf_rows) == 39
assert nsf_rows['Value'].between(0, 1).all()

nsf_color_map    = {'DSA-based': 'darkblue', 'hg38-based': 'orangered', 'chm13-based': 'olive'}
nsf_linetype_map = {'PacBio (40X)': 'solid', 'Short-read (100X)': 'dotted', 'Short-read hap-surj (100X)': 'dashdot'}
nsf_shape_map    = {'PacBio (40X)': 'o', 'Short-read (100X)': '^', 'Short-read hap-surj (100X)': '^'}

non_satellite_pr_plot = (ggplot(nsf_rows, aes(x='Ratio', y='Value', group='Series', color='Method', linetype='Modality', shape='Modality')) +
                         geom_line(size=0.5) +
                         geom_point(size=2) +
                         facet_grid('Metric ~ .') +
                         scale_color_manual(values=nsf_color_map) +
                         scale_linetype_manual(values=nsf_linetype_map) +
                         scale_shape_manual(values=nsf_shape_map) +
                         scale_x_discrete(labels={'T1N19': '1:19', 'T1N9': '1:9', 'T1N4': '1:4'}) +
                         scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.25, 0.25)) +
                         labs(title='Non-Satellite regions', x='Tumor:Normal Ratio', y='',
                              color='Reference', linetype='Modality', shape='Modality') +
                         theme_minimal() +
                         theme(
                             text=element_text(family='Arial'),
                             axis_text_x=element_text(rotation=0, hjust=0.5, color='black'),
                             axis_text_y=element_text(color='black'),
                             axis_title_x=element_text(color='black'),
                             axis_title_y=element_text(color='black'),
                             plot_title=element_text(color='black'),
                             strip_text=element_text(size=12, face='bold', color='black'),
                             legend_position='bottom',
                             legend_text=element_text(size=5))
                        )

ggsavefig_and_show(non_satellite_pr_plot, "Precision_Recall_Non-Satellite_DSA_hg38_chm13_20X_and_short-read", width=7, height=6)

#Ratio	Method	Modality	Metric	Value	Series
#T1N4	DSA-based	PacBio (40X)	Precision	0.702774	DSA-based / PacBio (40X)
#T1N4	DSA-based	PacBio (40X)	Recall	0.887760	DSA-based / PacBio (40X)
#T1N9	DSA-based	PacBio (40X)	Precision	0.637390	DSA-based / PacBio (40X)
#T1N9	DSA-based	PacBio (40X)	Recall	0.643342	DSA-based / PacBio (40X)
#T1N19	DSA-based	PacBio (40X)	Precision	0.492487	DSA-based / PacBio (40X)
#T1N19	DSA-based	PacBio (40X)	Recall	0.349094	DSA-based / PacBio (40X)
#T1N4	hg38-based	PacBio (40X)	Precision	0.657167	hg38-based / PacBio (40X)
#T1N4	hg38-based	PacBio (40X)	Recall	0.710640	hg38-based / PacBio (40X)
#T1N9	hg38-based	PacBio (40X)	Precision	0.574220	hg38-based / PacBio (40X)
#T1N9	hg38-based	PacBio (40X)	Recall	0.495987	hg38-based / PacBio (40X)
#T1N19	hg38-based	PacBio (40X)	Precision	0.407309	hg38-based / PacBio (40X)
#T1N19	hg38-based	PacBio (40X)	Recall	0.251183	hg38-based / PacBio (40X)
#T1N4	chm13-based	PacBio (40X)	Precision	0.363236	chm13-based / PacBio (40X)
#T1N4	chm13-based	PacBio (40X)	Recall	0.723529	chm13-based / PacBio (40X)
#T1N9	chm13-based	PacBio (40X)	Precision	0.285032	chm13-based / PacBio (40X)
#T1N9	chm13-based	PacBio (40X)	Recall	0.503756	chm13-based / PacBio (40X)
#T1N19	chm13-based	PacBio (40X)	Precision	0.167021	chm13-based / PacBio (40X)
#T1N19	chm13-based	PacBio (40X)	Recall	0.254733	chm13-based / PacBio (40X)
#T1N4	DSA-based	Short-read (100X)	Precision	0.511753	DSA-based / Short-read (100X)
#T1N4	DSA-based	Short-read (100X)	Recall	0.884904	DSA-based / Short-read (100X)
#T1N9	DSA-based	Short-read (100X)	Precision	0.523771	DSA-based / Short-read (100X)
#T1N9	DSA-based	Short-read (100X)	Recall	0.615302	DSA-based / Short-read (100X)
#T1N19	DSA-based	Short-read (100X)	Precision	0.437803	DSA-based / Short-read (100X)
#T1N19	DSA-based	Short-read (100X)	Recall	0.267545	DSA-based / Short-read (100X)
#T1N4	hg38-based	Short-read (100X)	Precision	0.811939	hg38-based / Short-read (100X)
#T1N4	hg38-based	Short-read (100X)	Recall	0.735156	hg38-based / Short-read (100X)
#T1N9	hg38-based	Short-read (100X)	Precision	0.764031	hg38-based / Short-read (100X)
#T1N9	hg38-based	Short-read (100X)	Recall	0.474866	hg38-based / Short-read (100X)
#T1N19	hg38-based	Short-read (100X)	Precision	0.514202	hg38-based / Short-read (100X)
#T1N19	hg38-based	Short-read (100X)	Recall	0.135059	hg38-based / Short-read (100X)
#T1N4	chm13-based	Short-read (100X)	Precision	0.643503	chm13-based / Short-read (100X)
#T1N4	chm13-based	Short-read (100X)	Recall	0.748971	chm13-based / Short-read (100X)
#T1N9	chm13-based	Short-read (100X)	Precision	0.594109	chm13-based / Short-read (100X)
#T1N9	chm13-based	Short-read (100X)	Recall	0.483639	chm13-based / Short-read (100X)
#T1N19	chm13-based	Short-read (100X)	Precision	0.335365	chm13-based / Short-read (100X)
#T1N19	chm13-based	Short-read (100X)	Recall	0.137297	chm13-based / Short-read (100X)
#T1N4	DSA-based	Short-read hap-surj (100X)	Precision	0.742055	DSA-based / Short-read hap-surj (100X)
#T1N9	DSA-based	Short-read hap-surj (100X)	Precision	0.704325	DSA-based / Short-read hap-surj (100X)
#T1N19	DSA-based	Short-read hap-surj (100X)	Precision	0.545443	DSA-based / Short-read hap-surj (100X)

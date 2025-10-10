#!/bin/bash
# mhsohny

#SBATCH --job-name=pbcpg
#SBATCH --account=stergachislab
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH -o logs/%x.%N.%j.slurm.out
#SBATCH -e logs/%x.%N.%j.slurm.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mhsohny@uw.edu
#SBATCH --export=ALL

#bamtocpg="/mmfs1/gscratch/stergachislab/mhsohny/Tools/pb-CpG-tools-v2.3.2-x86_64-unknown-linux-gnu/bin/aligned_bam_to_cpg_scores"
#pbcpgmodel="/mmfs1/gscratch/stergachislab/mhsohny/Tools/pb-CpG-tools-v2.3.2-x86_64-unknown-linux-gnu/models/pileup_calling_model.v1.tflite"
bamtocpg="/mmfs1/gscratch/stergachislab/mhsohny/Tools/pb-CpG-tools-v3.0.0-x86_64-unknown-linux-gnu/bin/aligned_bam_to_cpg_scores"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage   : _mCG_extraction_pbcpg_slurm.sh <input bam or cram> <sample prefix> <reference> [optional exclude bed]"
    echo "example : _mCG_extraction_pbcpg_slurm.sh COLO829BL_DSA_resetmapq.cram COLO829BL_DSA DSA_COLO829BL_v3.0.0.fasta Flagger.bed.gz" 
    exit 1
fi

set -euo pipefail

script="_mCG_extraction_pbcpg_slurm.sh"
echo "$(date '+%m-%d-%Y %H:%M:%S') sbatch $script $*" >> logs/job_submission_commandline.log # INFO: "$@" is arguments after the script that you put in

bcram="$1"
prefix="$2"
reference="$3"
exclude="$4"

    echo "Running CpG methylation extraction (pb-cpg-tool) for $2"

    ${bamtocpg} \
    --bam "${bcram}" \
    --ref "${reference}" \
    --output-prefix "${prefix}" \
    --pileup-mode model \
    --modsites-mode denovo \
    --min-coverage 4 \
    --min-mapq 1 \
    --hap-tag HP \
    --threads 99

    if [ -n "$exclude" ] && [ -f "$exclude" ]; then
        echo "Exclusion bed file provided: $exclude"
        echo "Subtracting exclude regions from mCG bed output..."
    
        bedtools subtract -a "${prefix}.combined.bed.gz" -b "$exclude" > "${prefix}.combined.filtered.bed"

        bgzip -@ 99 "${prefix}.combined.filtered.bed"
        tabix -p bed "${prefix}.combined.filtered.bed.gz"       
    fi

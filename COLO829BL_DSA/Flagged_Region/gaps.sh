dsa="DSA_COLO829BL_v3.0.0.fasta"

seqtk gap -l 1 $dsa \
| awk -F'\t' 'BEGIN {OFS="\t"} {print $0, $3-$2}' > DSA_COLO829BL_v3.0.0_gaps.bed
bgzip -@ 4 -f DSA_COLO829BL_v3.0.0_gaps.bed
tabix -p bed DSA_COLO829BL_v3.0.0_gaps.bed.gz

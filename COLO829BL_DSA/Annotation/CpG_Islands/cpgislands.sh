dsa="DSA_COLO829BL_v3.0.0.fasta"
dsa_sm_2bit="DSA_COLO829BL_v3.0.0_softmasked.2bit"
cpg_lh="ucsc-kent-tools-v455-2023-10-18/cpg_lh"
twoBitToFa="ucsc-kent-tools-v455-2023-10-18/twoBitToFa"
maskOutFa="ucsc-kent-tools-v455-2023-10-18/maskOutFa"

echo "Making CpG Islands tracks: https://genome.ucsc.edu/cgi-bin/hgTrackUi?g=cpgIslandExt"
echo "Unmasked Version:"
"$cpg_lh" "$dsa" \
    | awk '{
        $2 = $2 - 1;
        width = $3 - $2;
        printf("%s\t%d\t%s\t%s %s\t%s\t%s\t%0.0f\t%0.1f\t%s\t%s\n",
        $1, $2, $3, $5, $6, width, $6, width*$7*0.01, 100.0*2*$6/width, $7, $9);
    }' \
    | sort -k1,1 -k2,2n > DSA_COLO829BL_v3.0.0_CpGislands.bed

bgzip -@ 5 -f DSA_COLO829BL_v3.0.0_CpGislands.bed && \
        tabix -p bed DSA_COLO829BL_v3.0.0_CpGislands.bed.gz

awk 'BEGIN { OFS = "\t" } { print $1, $2, $3, "umCpGi_" NR "." $5 }' \
  DSA_COLO829BL_v3.0.0_CpGislands.bed \
  > DSA_COLO829BL_v3.0.0_CpGislands_id.bed

echo "Masked Version:"
"$twoBitToFa" "$dsa_sm_2bit" stdout \
        | "$maskOutFa" stdin hard stdout \
        | "$cpg_lh" /dev/stdin 2> cpg_lh.err \
    |  awk '{
         $2 = $2 - 1;
                 width = $3 - $2;
                 printf("%s\t%d\t%s\t%s %s\t%s\t%s\t%0.0f\t%0.1f\t%s\t%s\n", $1, $2, $3, $5, $6, width, $6, width*$7*0.01, 100.0*2*$6/width, $7, $9);
         }' \
    | sort -k1,1 -k2,2n > DSA_COLO829BL_v3.0.0_masked_CpGislands.bed

awk 'BEGIN { OFS = "\t" } { print $1, $2, $3, "CpGi_" NR "." $5 }' \
  DSA_COLO829BL_v3.0.0_masked_CpGislands.bed \
  > DSA_COLO829BL_v3.0.0_masked_CpGislands_id.bed

bgzip -@ 5 -f DSA_COLO829BL_v3.0.0_masked_CpGislands_id.bed && \
        tabix -p bed DSA_COLO829BL_v3.0.0_masked_CpGislands_id.bed.gz

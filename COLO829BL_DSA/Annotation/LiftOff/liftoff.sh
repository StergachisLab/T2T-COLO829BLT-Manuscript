gencode="/mmfs1/gscratch/stergachislab/mhsohny/Tools/Database/GFF/GENCODE/gencode.v47.annotation.gff3.gz"
reference="/mmfs1/gscratch/stergachislab/assemblies/hg38.analysisSet.fa"

target1="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_hap1.fasta"
output1="DSAv3.0.0_hap1_gencode.v47_LiftOff.gff3"

target2="/mmfs1/gscratch/stergachislab/mhsohny/SMaHT/DSA/DSA_COLO829BL_v3.0.0_hap2.fasta"
output2="DSAv3.0.0_hap2_gencode.v47_LiftOff.gff3"

echo "LiftOff GRCh38 GENCODE V47 to COLO829BL DSAv3.0.0 haplotype1"
  liftoff \
          -g "$gencode" \
          -o "$output1" \
          -u "unmapped_features.txt" \
          -copies \
          -sc 0.95 \
          -mm2_options="-a --end-bonus 5 --eqx -N 50 -p 0.5" \
          -polish \
          -cds \
          -exclude_partial \
          -p 79 \
          "$target1" \
          "$reference"

echo "Sorting gff3 for IGV Visualization"
  sort -k1,1 -k4,4n "$output1" > "${output1//.gff3/}_sorted.gff3"
  bgzip -@ 70 "${output1//.gff3/}_sorted.gff3" && tabix -p gff "${output1//.gff3/}_sorted.gff3.gz"

echo "LiftOff GRCh38 GENCODE V47 to COLO829BL DSAv3.0.0 haplotype2"
  liftoff \
          -g "$gencode" \
          -o "$output2" \
          -u "unmapped_features.txt" \
          -copies \
          -sc 0.95 \
          -mm2_options="-a --end-bonus 5 --eqx -N 50 -p 0.5" \
          -polish \
          -cds \
          -exclude_partial \
          -p 79 \
          "$target2" \
          "$reference"

echo "Sorting gff3 for IGV Visualization"
  sort -k1,1 -k4,4n "$output2" > "${output2//.gff3/}_sorted.gff3"
  bgzip -@ 70 "${output2//.gff3/}_sorted.gff3" && tabix -p gff "${output2//.gff3/}_sorted.gff3.gz"

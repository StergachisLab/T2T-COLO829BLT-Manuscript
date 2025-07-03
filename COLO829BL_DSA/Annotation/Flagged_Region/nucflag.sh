INBAM="COLO829BL_Fiber-seq_DATA_Aligned_onto_DSA_COLO829BL_v3.0.0_and_samtools_F2308"

mkdir -p plot
mkdir -p cov

nucflag \
  --infile "$INBAM" \
  --output_plot_dir ./plot \
  --output_cov_dir ./cov \
  --output_misasm DSA_COLO829BL_v3.0.0.NucFlag_F2308.bed \
  --output_status DSA_COLO829BL_v3.0.0.NucFlag_F2308_status.bed \
  --threads 94 \
  --processes 5

bgzip -f -@ 5 DSA_COLO829BL_v3.0.0.NucFlag_F2308.bed && \
  tabix -p bed DSA_COLO829BL_v3.0.0.NucFlag_F2308.bed.gz


# Codes used in the 1st round of revision

## Somatic SV analyses within centromeres
- `scan_Centromeric_unitsize_indels_SV.py` contains code to identify SVs within α-satellite, including unit-sized SVs in multiples of 171bp, the size of an α-satellite monomer.
- `colo829_centromere_genetic_analysis.py` contains exploratory data analysis code in terms of Figure 4C-I in the revised manuscript.
- `colo829_centromere_misc_analysis.py` contains miscellaneous analyses related to BEST analysis (Figure S1A-B), chromosomal assignment to the assembled contigs of the DSAs and initial exploratory analysis on the CDRs
- `colo829bl_cdr_alignment_karyoploteR.ipynb` contains code to generate Figure S1C 

## CDR analyses on multiple DSAs
- `colo829_centromere_epigenetic_analysis.ipynb` contains code analyzing data with respect to Figure 5B-P of the revised manuscript.

## Telomere analyses on COLO829BL/COLO829
- `colo829_telomere_counts.ipynb` and `colo829_telomere_restructuring.ipynb` contain codes to recapitulate analyses done to generate Figure 6 and related supplementary figures in the revised manuscript.

## DiMeLo-seq Data Added (BigWig files)
### COLO829BL CENPA DiMeLo-seq (ONT)
```
colo829b.dimelo.cpg.bw
colo829b.dimelo.m6a.bw
```

### COLOL829 CENPA DiMeLo-seq (ONT)
```
colo829t.dimelo.cpg.bw
colo829t.dimelo.m6a.bw
```
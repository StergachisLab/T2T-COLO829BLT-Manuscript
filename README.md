# A telomere-to-telomere map of somatic mutation burden and functional impact in cancer
Min-Hwan Sohn<sup>1,\*</sup>, Danilo Dubocanin<sup>2,\*</sup>, Mitchell R Vollger<sup>1,3</sup>, Youngjun Kwon<sup>3</sup>, Anna Minkina<sup>1</sup>, Katherine M Munson<sup>3</sup>, Samuel FM Hart<sup>3</sup>, Jane E Ranchalis<sup>1</sup>, Nancy L Parmalee<sup>4</sup>, Adriana E Sedeño-Cortés<sup>1</sup>, Jeffrey Ou<sup>4</sup>, Shane J. Neph<sup>1</sup>, Natalie YT Au<sup>4</sup>, Stephanie Bohaczuk<sup>1</sup>, Brianne Carroll<sup>3,5</sup>, Christian D Frazar<sup>3,5</sup>, William T Harvey<sup>3</sup>, Kendra Hoekzema<sup>3</sup>, Meng-Fan Huang<sup>3,5</sup>, Caitlin N Jacques<sup>3,5</sup>, Dana M Jensen<sup>4</sup>, J Thomas Kolar<sup>3,5</sup>, Rosa Lee<sup>2</sup>, Jiadong Lin<sup>3</sup>, Kelsey Loy<sup>4</sup>, Taralynn Mack<sup>3</sup>, Yizi Mao<sup>3</sup>, Matthew W. Mitchell<sup>6</sup>, Meranda M Pham<sup>4</sup>, Laura B. Scheinfeldt<sup>6</sup>, Gretchen Smith<sup>6</sup>, Erica Ryke<sup>3,5</sup>, Joshua D Smith<sup>3,5</sup>, Lila Sutherlin<sup>4</sup>, Elliott G Swanson<sup>1,3</sup>, Jeffrey M Weiss<sup>3,5</sup>, SMaHT Assembly WG, Claudia M. B. Carvalho<sup>7</sup>, Tim HH Coorens<sup>8,9</sup>, Kelley Harris<sup>3,10</sup>, Chia-Lin Wei<sup>3,5</sup>, Evan E Eichler<sup>3,11</sup>, Nicolas Altemose<sup>2,12</sup>, James T Bennett<sup>4</sup>, Andrew B Stergachis<sup>1,3,13,§</sup>

\*: contributed equally, §: corresponding author

1. Division of Medical Genetics, Department of Medicine, University of Washington, Seattle, WA, USA
2. Department of Genetics, School of Medicine, Stanford University, Stanford, CA, USA
3. Department of Genome Sciences, University of Washington School of Medicine, Seattle, WA, USA
4. Center for Developmental Biology and Regenerative Medicine, Seattle Children's Research Institute, Seattle, WA 98101, USA
5. The Northwest Genomics Center, University of Washington, Seattle, WA, USA
6. Coriell Institute for Medical Research, Camden, NJ, USA
7. Pacific Northwest Research Institute, Seattle, WA, USA
8. European Bioinformatics Institute, European Molecular Biology Laboratory (EMBL-EBI), Hinxton, UK
9. Broad Institute of MIT and Harvard, Cambridge MA, USA
10. Computational Biology Division, Fred Hutchinson Cancer Center, Seattle, WA, USA
11. Howard Hughes Medical Institute, University of Washington, Seattle, WA 98195, USA
12. Biohub, San Francisco, San Francisco, CA, USA
13. Brotman Baty Institute for Precision Medicine, Seattle, Washington, USA

Cancer genomes harbor extensive somatic genetic and epigenetic variation, yet large portions of the genome remain inaccessible to conventional reference-based analyses. Here, we demonstrate that pairing a near-telomere-to-telomere (T2T) diploid assembly of a donor with deep short- and long-read sequencing of their melanoma reveals that 16% of somatic variants localize to sequences absent from GRCh38, with satellite repeats acting as hotspots for UV-associated mutagenesis. Centromere kinetochore binding domains emerge as focal sites of chromosome arm aneuploidy breakpoints and somatic genetic variation, features we further demonstrate are also present in non-cancerous tissues. In contrast, extensive epigenetic remodeling of kinetochore binding domains appears specific to oncogenesis. Single-molecule telomere analyses reconstruct the cycle of telomere erosion and telomerase-mediated extension during tumor evolution, while haplotype-resolved chromatin maps reveal widespread somatic epimutations impacting regulatory elements. Together, these findings define previously inaccessible dimensions of somatic variation and establish a framework for telomere-to-telomere studies of human mosaicism.

Code used in analysis/figure generation as part of the T2T COLO829BLT manuscript: [Link](https://www.biorxiv.org/content/10.1101/2025.10.10.681725v1.full)

## Repeat Annotation on the Assembly
RepeatMasker, Tandem Repeat Finder (TRF) and DupMasker
Rhodonite: https://github.com/mrvollger/Rhodonite

## Assembly-to-assembly Alignment (Pair-wise Alignment)
https://github.com/mrvollger/asm-to-reference-alignment

## Short variants Calling (DeepVariant)
https://github.com/mrvollger/k-mer-variant-phasing

## Mutational Spectrum analysis
https://github.com/ryansohny/VCF2SPECTRUM

## Haplotype selective chromatin accessibility analysis
https://github.com/mrvollger/SMaHT-DSA-figures

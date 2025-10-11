# A telomere-to-telomere map of somatic mutation burden and functional impact in cancer
Min-Hwan Sohn<sup>1,\*</sup>, Danilo Dubocanin<sup>2,\*</sup>, Mitchell R Vollger<sup>1,3</sup>, Youngjun Kwon<sup>3</sup>, Anna Minkina<sup>1</sup>, Katherine M Munson<sup>3</sup>, Samuel FM Hart<sup>3</sup>, Jane E Ranchalis<sup>1</sup>, Nancy L Parmalee<sup>4</sup>, Adriana E Sedeño-Cortés<sup>1</sup>, Jeffrey Ou<sup>4</sup>, Natalie YT Au<sup>4</sup>, Stephanie Bohaczuk<sup>1</sup>, Brianne Carroll<sup>3,5</sup>, Christian D Frazar<sup>3,5</sup>, William T Harvey<sup>3</sup>, Kendra Hoekzema<sup>3</sup>, Meng-Fan Huang<sup>3,5</sup>, Caitlin N Jacques<sup>3,5</sup>, Dana M Jensen<sup>4</sup>, J Thomas Kolar<sup>3,5</sup>, Rosa Lee<sup>2</sup>, Jiadong Lin<sup>3</sup>, Kelsey Loy<sup>4</sup>, Taralynn Mack<sup>3</sup>, Yizi Mao<sup>3</sup>, Meranda M Pham<sup>4</sup>, Erica Ryke<sup>3,5</sup>, Joshua D Smith<sup>3,5</sup>, Lila Sutherlin<sup>4</sup>, Elliott G Swanson<sup>1,3</sup>, Jeffrey M Weiss<sup>3,5</sup>, SMaHT Assembly WG, Claudia Carvalho<sup>6</sup>, Tim HH Coorens<sup>7,8</sup>, Kelley Harris<sup>3,9</sup>, Chia-Lin Wei<sup>3,5</sup>, Evan E Eichler<sup>3,10</sup>, Nicolas Altemose<sup>2,11</sup>, James T Bennett<sup>4</sup>, Andrew B Stergachis<sup>1,3,12,§</sup>

\*: contributed equally, §: corresponding author

1. Division of Medical Genetics, Department of Medicine, University of Washington, Seattle, WA, USA
2. Department of Genetics, School of Medicine, Stanford University, Stanford, CA, USA
3. Department of Genome Sciences, University of Washington School of Medicine, Seattle, WA, USA
4. Center for Developmental Biology and Regenerative Medicine, Seattle Children's Research Institute, Seattle, WA 98101, USA
5. The Northwest Genomics Center, University of Washington, Seattle, WA, USA
6. Pacific Northwest research Institute, Seattle, WA, USA
7. European Bioinformatics Institute, European Molecular Biology Laboratory (EMBL-EBI), Hinxton, UK
8. Broad Institute of MIT and Harvard, Cambridge MA, USA
9. Computational Biology Division, Fred Hutchinson Cancer Center, Seattle, WA, USA
10. Howard Hughes Medical Institute, University of Washington, Seattle, WA 98195, USA
11. Chan Zuckerberg Biohub—San Francisco, San Francisco, CA, USA
12. Brotman Baty Institute for Precision Medicine, Seattle, Washington, USA

Oncogenesis involves widespread genetic and epigenetic alterations, yet the full spectrum of somatic variation genome-wide remains unresolved. We generated a near-telomere-to-telomere (T2T) diploid assembly of a donor paired with deep short- and long-read sequencing of their melanoma. This revealed that 16% of somatic variants occur in sequences absent from GRCh38, with satellite repeats acting as hotspots for UV-induced damage due to sequence-intrinsic mutability and inefficient repair. Centromere kinetochore domains emerged as focal sites of structural, genetic, and epigenetic variation, leading to remodeling of centromere kinetochore binding domains during tumor evolution. Single-molecule telomere reconstructions uncovered cycles of attrition, deletion, and telomerase-mediated extension that shape cancer telomeres. Finally, diploid chromatin maps exposed that copy number alterations and epimutations, rather than point mutations, predominate in rewiring cancer regulatory programs. These findings define the full landscape of a cancer’s somatic variation and their functional impact, establishing a blueprint for T2T studies of mosaicism.

Code used in analysis/figure generation as part of the T2T COLO829BLT manuscript: [link here when it's up in biorxiv]

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


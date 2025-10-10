# A telomere-to-telomere map of somatic mutation burden and functional impact in cancer
Min-Hwan Sohn<sup>1,*</sup>, Danilo Dubocanin<sup>2,\*</sup>, Mitchell R Vollger1,3, Youngjun Kwon3, Anna Minkina1, Katherine M Munson3, Samuel FM Hart3, Jane E Ranchalis1, Nancy L Parmalee4, Adriana E Sedeño-Cortés1, Jeffrey Ou4, Natalie YT Au4, Stephanie Bohaczuk1, Brianne Carroll3,5, Christian D Frazar3,5, William T Harvey3, Kendra Hoekzema3, Meng-Fan Huang3,5, Caitlin N Jacques3,5, Dana M Jensen4, J Thomas Kolar3,5, Rosa Lee2, Jiadong Lin3, Kelsey Loy4, Taralynn Mack3, Yizi Mao3, Meranda M Pham4, Erica Ryke3,5, Joshua D Smith3,5, Lila Sutherlin4, Elliott G Swanson3,1, Jeffrey M Weiss3,5, SMaHT Assembly WG, Claudia  Carvalho6, Tim HH Coorens7,8, Kelley Harris3,9, Chia-Lin Wei3,5, Evan E Eichler3,10, Nicolas Altemose2,11, James T Bennett4, Andrew B Stergachis1,3,12

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

## Assembly to Assembly Alignment (Pair-wise Alignment Format (PAF) Generation)
https://github.com/mrvollger/asm-to-reference-alignment

## SNV/Indel Calling
https://github.com/mrvollger/k-mer-variant-phasing

## Mutational Spectrum analysis
https://github.com/ryansohny/VCF2SPECTRUM


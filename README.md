# A telomere-to-telomere map of somatic mutation burden and functional impact in cancer
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


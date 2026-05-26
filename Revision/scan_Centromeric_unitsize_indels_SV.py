# INFO: ############################################## 
# Scan BAM/CRAM files for unit-sized indels.
# Author: Sohny (Min-Hwan Sohn)
#####################################################

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import median
import sys
import os
import gzip
import tempfile
import pysam
import pyranges as pr
import pandas as pd
import defopt
from tqdm import tqdm


@dataclass
class IndelRecord:
    """Represents a single indel event."""
    chrom: str
    start: int  # NOTE: 0-based
    end: int    # NOTE: 0-based, exclusive (BED format)
    indel_type: str  # NOTE: "INS" or "DEL"
    indel_length: int
    unit_multiple: float  # NOTE: x times unit size
    is_unit_sized: bool  # NOTE: whether indel is within tolerance of unit size
    median_bq: str  # NOTE: median base quality or "." for deletions
    sequence: str
    read_name: str
    is_primary: bool
    gc_identity: float  # NOTE: gap-compressed identity: 1 - (NM - gap_sum + gap_opens) / (M_sum + gap_opens)
    aligned_fraction: float  # NOTE: aligned bases / total query length (accounts for soft clips)
    read_length: int  # NOTE: length of the query sequence (read.query_length)
    alignment_span: int  # NOTE: reference span of the alignment (reference_end - reference_start)
    min_dist_query: int  # NOTE: min distance from indel to read ends (query coordinates)
    min_dist_ref: int  # NOTE: min distance from indel to alignment ends (reference coordinates)
    overlap_class: str = "."  # NOTE: "target", "non_target", or "." if not applicable

    @property
    def indel_id(self) -> str:
        """Generate indel ID as chrom:start-end string."""
        return f"{self.chrom}:{self.start}-{self.end}"

    def to_bed_line(self) -> str:
        """Convert to BED format line."""
        return "\t".join([
            self.chrom,
            str(self.start),
            str(self.end),
            self.indel_id,
            self.indel_type,
            str(self.indel_length),
            f"{self.unit_multiple:.2f}",
            str(self.is_unit_sized),
            self.median_bq,
            self.overlap_class,
            f"{self.gc_identity:.4f}",
            f"{self.aligned_fraction:.4f}",
            str(self.read_length),
            str(self.alignment_span),
            str(self.min_dist_query),
            str(self.min_dist_ref),
            self.read_name,
            str(self.is_primary),
            self.sequence
        ])


def calculate_median_bq(qualities: List[int]) -> str:
    """Calculate median base quality from a list of quality scores."""
    if not qualities:
        return "."
    return f"{median(qualities):.1f}"


def calculate_read_metrics(read: pysam.AlignedSegment) -> Tuple[float, float]:
    """
    Calculate gap-compressed identity and aligned fraction for a read.

    Gap-compressed identity formula (Heng Li's definition):
        gc_identity = 1 - (NM - gap_sum + gap_opens) / (M_sum + gap_opens)

    Where:
        - NM: edit distance (mismatches + insertions + deletions)
        - gap_sum: total length of all gaps (I + D bases)
        - gap_opens: number of gap opening events (each I/D operation counts as 1)
        - M_sum: total aligned bases (M/=/X operations)

    Reference: https://lh3.github.io/2018/11/25/on-the-definition-of-sequence-identity

    Aligned fraction: aligned_bases / total_query_length
    - Accounts for soft-clipped bases

    Args:
        read: pysam AlignedSegment object

    Returns:
        Tuple of (gc_identity, aligned_fraction)
    """
    if read.cigartuples is None:
        return 0.0, 0.0

    # INFO: Count CIGAR operations (both lengths and number of events)
    cigar_lengths = {op: 0 for op in range(9)}  # ops 0-8: total bases
    cigar_opens = {op: 0 for op in range(9)}  # ops 0-8: number of events

    for op, length in read.cigartuples:
        cigar_lengths[op] += length
        cigar_opens[op] += 1

    # NOTE: CIGAR ops: 0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 6=P, 7==, 8=X
    m_bases = cigar_lengths[0]  # M (match or mismatch, ambiguous)
    matches = cigar_lengths[7]  # = (sequence match)
    mismatches = cigar_lengths[8]  # X (sequence mismatch)
    insertions = cigar_lengths[1]  # I total bases
    deletions = cigar_lengths[2]  # D total bases
    soft_clips = cigar_lengths[4]  # S

    # INFO: Gap statistics
    gap_sum = insertions + deletions  # NOTE: Total gap bases
    gap_opens = cigar_opens[1] + cigar_opens[2]  # NOTE: Number of gap events

    # INFO: M_sum = aligned bases (M + = + X)
    m_sum = m_bases + matches + mismatches

    # INFO: Calculate aligned fraction
    # NOTE: Total query length = M + I + S + = + X (bases that consume query)
    total_query = m_bases + insertions + soft_clips + matches + mismatches
    aligned_query = m_bases + insertions + matches + mismatches

    if total_query > 0:
        aligned_fraction = aligned_query / total_query
    else:
        aligned_fraction = 0.0

    # INFO: Calculate gap-compressed identity
    # NOTE: 1 - (NM - gap_sum + gap_opens) / (M_sum + gap_opens)
    # NOTE: We need NM tag for edit distance
    try:
        nm = read.get_tag('NM')
    except KeyError:
        # NOTE: Probably won't happen (All have NM tags availalble)
        if matches > 0 or mismatches > 0:
            # NOTE:: NM = mismatches + insertions + deletions
            nm = mismatches + insertions + deletions
        else:
            # NOTE:: Cannot calculate accurately without NM tag
            return 0.0, aligned_fraction

    denominator = m_sum + gap_opens
    if denominator > 0:
        numerator = nm - gap_sum + gap_opens
        gc_identity = 1 - (numerator / denominator)
        # Clamp to [0, 1] range
        gc_identity = max(0.0, min(1.0, gc_identity))
    else:
        gc_identity = 0.0

    return gc_identity, aligned_fraction


def is_unit_sized(indel_length: int, unit_size: int, tolerance: float) -> Tuple[bool, float]:
    """
    Check if an indel length is approximately a multiple of unit_size.

    Args:
        indel_length: Length of the indel
        unit_size: Expected unit size (e.g., 171 for alpha-satellite)
        tolerance: Fraction tolerance (e.g., 0.99 means allow 1% deviation)

    Returns:
        Tuple of (is_unit_sized, unit_multiple)
    """
    if unit_size <= 0:
        return False, 0.0

    unit_multiple = indel_length / unit_size

    # INFO: Round to nearest integer to check deviation
    nearest_int = round(unit_multiple)
    if nearest_int == 0:
        return False, unit_multiple

    ratio = unit_multiple / nearest_int

    # tolerance of 0.99 means we allow (1 - 0.99) = 1% deviation on each side
    deviation_allowed = 1 - tolerance
    lower_bound = 1 - deviation_allowed
    upper_bound = 1 + deviation_allowed

    is_match = lower_bound <= ratio <= upper_bound

    return is_match, unit_multiple


def collapse_by_contig(pyr: pr.PyRanges) -> pr.PyRanges:
    """
    Collapse all intervals on the same chromosome into a single interval.

    For each chromosome, takes the minimum start and maximum end to create
    one interval spanning all original intervals on that chromosome.

    Args:
        pyr: PyRanges object

    Returns:
        PyRanges object with one interval per chromosome
    """
    df = pyr.copy()
    collapsed = df.groupby('Chromosome').agg({
        'Start': 'min',
        'End': 'max'
    }).reset_index()
    return pr.PyRanges(collapsed)


def make_windows(
    total_bed: str,
    target_bed: str,
    window_size: int,
    collapse_target: bool = False,
    overlap_fraction: float = 0.5,
    padding: int = 50000
) -> pr.PyRanges:
    """
    Create windows from total_bed and label them based on overlap with target_bed.

    1. Reads the total interval BED (larger regions containing targets)
    2. Creates windows of specified size within total intervals
    3. Labels each window as "target" or "non_target" based on overlap fraction
    4. Returns all windows with an "overlap_class" column

    Args:
        total_bed: Path to BED file with total intervals (superset containing targets)
        target_bed: Path to BED file with target intervals (e.g., CDR regions)
        window_size: Size of windows to create within total intervals
        collapse_target: Whether to collapse target intervals by contig (merge all
                         intervals on the same chromosome into one)
        overlap_fraction: Minimum fraction of window that must overlap with target
                          to be classified as "target" (default: 0.5 = >50%)
        padding: Padding in bp to extend target intervals on each side (default: 50000)

    Returns:
        PyRanges object with all windows and "overlap_class" column ("target" or "non_target")
    """
    total_pr = pr.read_bed(total_bed)
    target_pr = pr.read_bed(target_bed)

    # INFO: Pad target intervals and merge any resulting overlaps
    if padding > 0:
        target_pr = target_pr.extend_ranges(ext=padding, use_strand=False)
    target_pr = target_pr.merge_overlaps()

    if collapse_target:
        target_pr = collapse_by_contig(target_pr)

    # INFO: Check if window size is valid
    if total_pr.lengths().min() < window_size:
        raise ValueError(
            f"Window size {window_size} is larger than the smallest interval "
            f"in total_bed ({total_pr.lengths().min()} bp)"
        )

    total_windows_pr = total_pr.window_ranges(window_size)

    # NOTE: Get window dataframe
    windows_df = total_windows_pr.copy().reset_index(drop=True)
    windows_df['window_idx'] = range(len(windows_df))

    # NOTE: Get target dataframe
    targets_df = target_pr.copy().reset_index(drop=True)

    # NOTE:: Calculate overlap for each window manually
    overlap_lengths = []

    for _, window in tqdm(windows_df.iterrows(), total=len(windows_df), desc="Classifying windows", file=sys.stderr):
        w_chrom = window['Chromosome']
        w_start = window['Start']
        w_end = window['End']

        # NOTE: Find overlapping targets on same chromosome
        chrom_targets = targets_df[targets_df['Chromosome'] == w_chrom]

        total_overlap = 0
        for _, target in chrom_targets.iterrows():
            t_start = target['Start']
            t_end = target['End']

            # NOTE: Calculate overlap
            overlap_start = max(w_start, t_start)
            overlap_end = min(w_end, t_end)
            overlap_len = max(0, overlap_end - overlap_start)
            total_overlap += overlap_len

        overlap_lengths.append(total_overlap)

    windows_df['overlap_len'] = overlap_lengths
    windows_df['window_size'] = windows_df['End'] - windows_df['Start']
    windows_df['overlap_frac'] = windows_df['overlap_len'] / windows_df['window_size']

    # INFO: Classify windows based on overlap fraction threshold
    windows_df['overlap_class'] = windows_df['overlap_frac'].apply(
        lambda x: 'target' if x > overlap_fraction else 'non_target'
    )

    # INFO: Create final PyRanges with classification
    result_df = windows_df[['Chromosome', 'Start', 'End', 'overlap_class']].copy()
    result_pr = pr.PyRanges(result_df)

    return result_pr


def parse_cigar_for_indels(
    read: pysam.AlignedSegment,
    unit_size: int,
    tolerance: float,
    min_indel_size: int = 10,
    ref_fasta: Optional[pysam.FastaFile] = None
) -> List[IndelRecord]:
    """
    Parse CIGAR string to extract indels that match the unit size criteria.

    Args:
        read: pysam AlignedSegment object
        unit_size: Expected unit size for filtering
        tolerance: Tolerance for unit size matching
        min_indel_size: Minimum indel size to report (default 10bp)
        ref_fasta: Optional pysam.FastaFile for fetching deleted sequences

    Returns:
        List of IndelRecord objects
    """
    if read.cigartuples is None:
        return []

    indels = []

    # INFO: Calculate read-level metrics once
    gc_identity, aligned_fraction = calculate_read_metrics(read)

    # INFO: Get alignment boundaries for min_dist calculation
    align_start = read.reference_start  # 0-based, left-most aligned position
    align_end = read.reference_end  # 0-based, right-most aligned position (exclusive)

    # INFO: Track positions
    # NOTE: ref_pos: 0-based reference position (pysam uses 0-based)
    # NOTE: query_pos: position in query/read sequence (0-based)
    ref_pos = read.reference_start  # NOTE: 0-based in pysam  
    query_pos = 0

    chrom = read.reference_name
    read_name = read.query_name
    is_primary = not read.is_supplementary and not read.is_secondary

    query_sequence = read.query_sequence
    query_qualities = read.query_qualities
    query_length = read.query_length
    alignment_span = read.reference_length  # NOTE: reference_end - reference_start

    # INFO: CIGAR operations (OP;op):
    """
    M/=/X (0,7,8): consume both ref and query
    I (1): consume query only (insertion to reference)
    D (2): consume ref only (deletion from reference)
    N (3): consume ref only (skipped region)
    S (4): consume query only (soft clip)
    H (5): consume neither (hard clip)
    P (6): consume neither (padding)
    """

    for op, length in read.cigartuples:
        if op == 1:  # NOTE: Insertion
            # INFO: Report all indels >= min_indel_size
            if length >= min_indel_size:
                is_match, unit_multiple = is_unit_sized(length, unit_size, tolerance)

                # NOTE: For insertion: position is between ref_pos-1 and ref_pos (0-based)
                # NOTE: In BED format, we represent this as a point: [ref_pos, ref_pos+1)
                # NOTE: But conventionally for insertions, we use [ref_pos, ref_pos] as 0-width
                # NOTE: Here we'll use [ref_pos, ref_pos] to indicate insertion point
                start = ref_pos  # NOTE: 0-based
                end = ref_pos    # NOTE: 0-based, same as start for insertion point

                # INFO: Extract inserted sequence
                ins_seq = query_sequence[query_pos:query_pos + length] if query_sequence else "."

                # INFO: Calculate median base quality of inserted bases
                if query_qualities is not None:
                    ins_quals = query_qualities[query_pos:query_pos + length]
                    med_bq = calculate_median_bq(list(ins_quals))
                else:
                    med_bq = "."

                # INFO: Calculate min distance from indel to alignment ends (reference coords)
                dist_to_left_ref = ref_pos - align_start
                dist_to_right_ref = align_end - ref_pos
                min_dist_ref = min(dist_to_left_ref, dist_to_right_ref)

                # INFO: Calculate min distance from indel to read ends (query coords)
                dist_to_left_query = query_pos
                dist_to_right_query = query_length - (query_pos + length)
                min_dist_query = min(dist_to_left_query, dist_to_right_query)

                indels.append(IndelRecord(
                    chrom=chrom,
                    start=start,
                    end=end,
                    indel_type="INS",
                    indel_length=length,
                    unit_multiple=unit_multiple,
                    is_unit_sized=is_match,
                    median_bq=med_bq,
                    sequence=ins_seq,
                    read_name=read_name,
                    is_primary=is_primary,
                    gc_identity=gc_identity,
                    aligned_fraction=aligned_fraction,
                    read_length=query_length,
                    alignment_span=alignment_span,
                    min_dist_query=min_dist_query,
                    min_dist_ref=min_dist_ref
                ))

            query_pos += length

        elif op == 2:  # NOTE: Deletion
            # INFO: Report all indels >= min_indel_size
            if length >= min_indel_size:
                is_match, unit_multiple = is_unit_sized(length, unit_size, tolerance)

                # NOTE: For deletion: the deleted region spans [ref_pos, ref_pos + length)
                start = ref_pos  # NOTE: 0-based
                end = ref_pos + length  # NOTE: 0-based, exclusive

                # INFO: Fetch deleted sequence from reference if available
                if ref_fasta is not None:
                    try:
                        del_seq = ref_fasta.fetch(chrom, start, end)
                    except (KeyError, ValueError):
                        del_seq = f"N*{length}"
                else:
                    del_seq = f"N*{length}"

                # INFO: Calculate min distance from indel to alignment ends (reference coords)
                # NOTE:For deletions, use the midpoint of the deleted region
                indel_midpoint = ref_pos + length // 2
                dist_to_left_ref = indel_midpoint - align_start
                dist_to_right_ref = align_end - indel_midpoint
                min_dist_ref = min(dist_to_left_ref, dist_to_right_ref)

                # INFO: Calculate min distance from indel to read ends (query coords)
                # NOTE: Deletion doesn't consume query, so query_pos is at the deletion point
                dist_to_left_query = query_pos
                dist_to_right_query = query_length - query_pos
                min_dist_query = min(dist_to_left_query, dist_to_right_query)

                indels.append(IndelRecord(
                    chrom=chrom,
                    start=start,
                    end=end,
                    indel_type="DEL",
                    indel_length=length,
                    unit_multiple=unit_multiple,
                    is_unit_sized=is_match,
                    median_bq=".",  # No base quality for deletions
                    sequence=del_seq,
                    read_name=read_name,
                    is_primary=is_primary,
                    gc_identity=gc_identity,
                    aligned_fraction=aligned_fraction,
                    read_length=query_length,
                    alignment_span=alignment_span,
                    min_dist_query=min_dist_query,
                    min_dist_ref=min_dist_ref
                ))

            ref_pos += length

        elif op in (0, 7, 8):  # M, =, X - consume both
            ref_pos += length
            query_pos += length

        elif op == 3:  # N - consume ref only
            ref_pos += length

        elif op == 4:  # S - consume query only
            query_pos += length

        # H (5) and P (6) consume neither

    return indels


def get_regions_from_pyranges(pyr: pr.PyRanges) -> List[Tuple[str, int, int]]:
    """
    Extract regions from a PyRanges object as list of tuples.

    Args:
        pyr: PyRanges object

    Returns:
        List of (chrom, start, end) tuples
    """
    regions = []
    for _, row in pyr.copy().iterrows():
        regions.append((row['Chromosome'], row['Start'], row['End']))
    return regions


def filter_indels_by_regions(
    indels: List[IndelRecord],
    target_pr: pr.PyRanges
) -> List[IndelRecord]:
    """
    Filter indels to only those overlapping target regions using PyRanges.

    Args:
        indels: List of IndelRecord objects
        target_pr: PyRanges object with target regions

    Returns:
        Filtered list of IndelRecord objects
    """
    if not indels:
        return []

    indel_df = pd.DataFrame({
        'Chromosome': [i.chrom for i in indels],
        'Start': [i.start for i in indels],
        'End': [max(i.end, i.start + 1) for i in indels],  # Ensure at least 1bp for overlap
        'idx': list(range(len(indels)))
    })

    indel_pr = pr.PyRanges(indel_df)

    # NOTE: Find overlaps
    overlapped = indel_pr.overlap(target_pr)

    if overlapped.empty:
        return []

    # NOTE: Get indices of overlapping indels
    kept_indices = set(overlapped.copy()['idx'].tolist())

    return [indels[i] for i in sorted(kept_indices)]


def annotate_indels_with_overlap_class(
    indels: List[IndelRecord],
    windows_pr: pr.PyRanges
) -> List[IndelRecord]:
    """
    Annotate indels with their overlap_class based on which window they fall into.

    Args:
        indels: List of IndelRecord objects
        windows_pr: PyRanges object with windows containing 'overlap_class' column

    Returns:
        List of IndelRecord objects with overlap_class field set
    """
    if not indels:
        return []

    indel_df = pd.DataFrame({
        'Chromosome': [i.chrom for i in indels],
        'Start': [i.start for i in indels],
        'End': [max(i.end, i.start + 1) for i in indels],  # Ensure at least 1bp for overlap
        'idx': list(range(len(indels)))
    })

    indel_pr = pr.PyRanges(indel_df)


    joined = indel_pr.join_overlaps(windows_pr, join_type="left", suffix="_w")

    if joined.empty:
        return indels

    df = joined.copy()

    # INFO: Create a mapping from indel index to overlap_class
    idx_to_class = {}
    for _, row in df.iterrows():
        idx = row['idx']
        if idx not in idx_to_class:
            overlap_class = row.get('overlap_class', row.get('overlap_class_w', '.'))
            if pd.isna(overlap_class):
                overlap_class = '.'
            idx_to_class[idx] = overlap_class

    annotated_indels = []
    for i, indel in enumerate(indels):
        indel.overlap_class = idx_to_class.get(i, '.')
        annotated_indels.append(indel)

    return annotated_indels


def extract_reads(
    input_bcram: str,
    read_names: set,
    regions: Optional[List[str]] = None,
    reference: Optional[str] = None,
    bam_output: Optional[str] = None,
    fasta_output: Optional[str] = None,
    threads: int = 1
) -> None:
    """
    Extract reads with unit-sized indels to BAM and/or FASTA.

    Uses samtools view -N for C-speed filtering instead of Python iteration.

    Args:
        input_bcram: Path to input BAM/CRAM file
        read_names: Set of read names to extract
        regions: Optional list of region strings to scan (None = whole file)
        reference: Path to reference FASTA (for CRAM)
        bam_output: Path to output BAM/CRAM file (None = skip)
        fasta_output: Path to output FASTA file (None = skip)
        threads: Number of threads for compression/sorting
    """
    if not read_names or (bam_output is None and fasta_output is None):
        return

    print(f"Extracting {len(read_names)} reads with unit-sized indels...", file=sys.stderr)

    is_cram_out = bam_output and bam_output.endswith('.cram')

    namelist_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, prefix='readnames_'
    )
    try:
        for name in read_names:
            namelist_tmp.write(name + '\n')
        namelist_tmp.close()

        if bam_output:
            unsorted_tmp = bam_output + ".unsorted.tmp"
            view_args = [
                "-N", namelist_tmp.name,
                "-@", str(threads),
                "-o", unsorted_tmp,
            ]
            if is_cram_out:
                view_args += [
                    "-O", "cram",
                    "--output-fmt-option", "store_md=1",
                    "--output-fmt-option", "store_nm=1",
                    "--output-fmt-option", "embed_ref=1",
                ]
            else:
                view_args += ["-O", "bam"]
            if reference:
                view_args += ["--reference", reference]
            view_args.append(input_bcram)
            pysam.view(*view_args, catch_stdout=False)

            ext = ".cram" if is_cram_out else ".bam"
            sorted_tmp = bam_output + ".sort.tmp" + ext
            sort_args = ["-o", sorted_tmp, "-@", str(threads)]
            if is_cram_out:
                sort_args += [
                    "-O", "cram",
                    "--output-fmt-option", "store_md=1",
                    "--output-fmt-option", "store_nm=1",
                    "--output-fmt-option", "embed_ref=1",
                ]
                if reference:
                    sort_args += ["--reference", reference]
            else:
                sort_args += ["-O", "bam"]
            sort_args.append(unsorted_tmp)
            pysam.sort(*sort_args)
            os.replace(sorted_tmp, bam_output)
            os.remove(unsorted_tmp)
            pysam.index(bam_output)
            print(f"Wrote reads to {bam_output} (sorted + indexed)", file=sys.stderr)

        # INFO: FASTA: derive from the output BAM/CRAM (small file) or filter input
        if fasta_output:
            source = bam_output if bam_output else input_bcram
            if source == input_bcram:
                # NOTE: No BAM output requested; filter input directly with view
                source_tmp = tempfile.NamedTemporaryFile(
                    suffix='.bam', delete=False, prefix='fasta_src_'
                ).name
                fa_view_args = [
                    "-N", namelist_tmp.name,
                    "-@", str(threads),
                    "-O", "bam",
                    "-o", source_tmp,
                ]
                if reference:
                    fa_view_args += ["--reference", reference]
                fa_view_args.append(input_bcram)
                pysam.view(*fa_view_args, catch_stdout=False)
                source = source_tmp
            else:
                source_tmp = None

            # INFO: Read the (small) filtered file to build FASTA
            in_mode = 'rc' if source.endswith('.cram') else 'rb'
            in_kwargs = {'reference_filename': reference} if source.endswith('.cram') and reference else {}
            fasta_reads = {}
            with pysam.AlignmentFile(source, in_mode, **in_kwargs) as f:
                for read in f.fetch(until_eof=True):
                    if read.is_unmapped:
                        continue
                    name = read.query_name
                    is_primary = not read.is_supplementary and not read.is_secondary
                    if name not in fasta_reads or (is_primary and not fasta_reads[name][0]):
                        fasta_reads[name] = (is_primary, read.query_sequence)

            if source_tmp:
                os.remove(source_tmp)

            open_func = gzip.open if fasta_output.endswith('.gz') else open
            write_mode = 'wt' if fasta_output.endswith('.gz') else 'w'
            with open_func(fasta_output, write_mode) as f:
                for name, (_, seq) in fasta_reads.items():
                    if seq:
                        f.write(f">{name}\n{seq}\n")
            print(f"Wrote {len(fasta_reads)} reads to {fasta_output}", file=sys.stderr)
    finally:
        os.remove(namelist_tmp.name)


def process_region(
    bam_path: str,
    region: str,
    unit_size: int,
    tolerance: float,
    min_indel_size: int = 10,
    reference: Optional[str] = None
) -> List[IndelRecord]:
    """
    Process a genomic region to find indels meeting size criteria.

    Args:
        bam_path: Path to BAM/CRAM file
        region: Genomic region string (e.g., "chr1:1000-2000")
        unit_size: Unit size for indel filtering
        tolerance: Tolerance for unit size matching
        min_indel_size: Minimum indel size to report (default 10bp)
        reference: Path to reference FASTA (required for CRAM, optional for fetching deleted sequences)

    Returns:
        List of IndelRecord objects
    """
    # INFO: Determine file type and open appropriately
    if bam_path.endswith('.cram'):
        mode = 'rc'
        kwargs = {'reference_filename': reference} if reference else {}
    else:
        mode = 'rb'
        kwargs = {}

    indels = []
    unit_read_names = set()

    # INFO: Open reference FASTA if provided (for fetching deleted sequences)
    ref_fasta = None
    if reference:
        try:
            ref_fasta = pysam.FastaFile(reference)
        except Exception:
            ref_fasta = None

    try:
        with pysam.AlignmentFile(bam_path, mode, **kwargs) as samfile:
            # INFO: Fetch reads from region
            iterator = samfile.fetch(region=region)

            for read in iterator:
                # NOTE: Skip unmapped reads
                if read.is_unmapped:
                    continue

                read_indels = parse_cigar_for_indels(read, unit_size, tolerance, min_indel_size, ref_fasta)
                indels.extend(read_indels)
                if any(ind.is_unit_sized for ind in read_indels):
                    unit_read_names.add(read.query_name)
    finally:
        if ref_fasta is not None:
            ref_fasta.close()

    return indels, unit_read_names


def process_region_wrapper(args: Tuple) -> List[IndelRecord]:
    """Wrapper for multiprocessing."""
    return process_region(*args)


def scan_unitsize_indels(
    input_bcram: str,
    output_bed: str,
    unit_size: int,
    tolerance: float = 0.99,
    min_indel_size: int = 10,
    threads: int = 1,
    total_bed: Optional[str] = None,
    target_bed: Optional[str] = None,
    window_size: int = 1000,
    collapse_target: bool = False,
    padding: int = 50000,
    reference: Optional[str] = None,
    gzip_output: bool = False,
    reads_output: Optional[str] = None,
    fasta_output: Optional[str] = None
) -> None:
    """
    Main function to scan BAM/CRAM for indels meeting size criteria.

    Args:
        input_bcram: Path to input BAM/CRAM file
        output_bed: Path to output BED file (use "-" for stdout)
        unit_size: Unit size for indel filtering (e.g., 171 for alpha-satellite)
        tolerance: Tolerance for unit size matching (0.99 = allow 1% deviation)
        min_indel_size: Minimum indel size to report (default 10bp)
        threads: Number of threads for parallel processing
        total_bed: BED file with total intervals (superset containing target regions)
        target_bed: BED file with target regions (e.g., CDRs) to restrict analysis
        window_size: Window size for processing within total intervals
        collapse_target: Whether to collapse target intervals by contig
        padding: Padding in bp to extend target intervals on each side (default: 50000)
        reference: Path to reference FASTA (required for CRAM)
        gzip_output: Whether to gzip the output
    """

    # INFO: Header for BED file
    header = "#chrom\tstart\tend\tindel_id\tindel_type\tindel_length\tunit_multiple\tis_unit_sized\tmedian_bq\toverlap_class\tgc_identity\taligned_fraction\tread_length\talignment_span\tmin_dist_query\tmin_dist_ref\tread_name\tis_primary\tsequence"

    all_indels = []
    all_read_names = set()
    scan_regions = None
    target_pr = None
    windows_pr = None

    if target_bed:
        # INFO: Load target regions, pad, and merge overlaps
        target_pr = pr.read_bed(target_bed)
        if padding > 0:
            target_pr = target_pr.extend_ranges(ext=padding, use_strand=False)
        target_pr = target_pr.merge_overlaps()
        if collapse_target:
            target_pr = collapse_by_contig(target_pr)

        if total_bed:
            # INFO: Use windowing approach with total_bed and target_bed
            # NOTE: This creates windows from total_bed and labels them as "target" or "non_target"
            windows_pr = make_windows(total_bed, target_bed, window_size, collapse_target, padding=padding)
            regions = get_regions_from_pyranges(windows_pr)
        else:
            # INFO: Use target_bed directly without windowing
            regions = get_regions_from_pyranges(target_pr)

        # INFO: Convert regions to pysam-compatible format
        region_strings = [f"{chrom}:{start}-{end}" for chrom, start, end in regions]
        scan_regions = region_strings

        if threads > 1 and len(region_strings) > 1:
            # INFO: Parallel processing by region
            tasks = [
                (input_bcram, region, unit_size, tolerance, min_indel_size, reference)
                for region in region_strings
            ]

            with ProcessPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(process_region_wrapper, task): task[1] for task in tasks}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing regions", file=sys.stderr):
                    region = futures[future]
                    try:
                        indels, read_names = future.result()
                        all_indels.extend(indels)
                        all_read_names.update(read_names)
                    except Exception as e:
                        print(f"Error processing {region}: {e}", file=sys.stderr)
        else:
            # INFO: Single-threaded processing of all regions
            for region in tqdm(region_strings, desc="Processing regions", file=sys.stderr):
                indels, read_names = process_region(input_bcram, region, unit_size, tolerance, min_indel_size, reference)
                all_indels.extend(indels)
                all_read_names.update(read_names)

        if total_bed and windows_pr is not None:
            # INFO: Annotate indels with overlap_class based on window classification
            # NOTE: Filter to only indels within windows first
            all_indels = filter_indels_by_regions(all_indels, windows_pr)
            all_indels = annotate_indels_with_overlap_class(all_indels, windows_pr)
        else:
            # INFO: Filter indels to only those within target regions
            # NOTE: This is needed because fetch() returns reads overlapping the region,
            # NOTE: but indels might be outside the exact target boundaries
            all_indels = filter_indels_by_regions(all_indels, target_pr)

    else:
        # INFO: No target BED - process entire BAM/CRAM
        if input_bcram.endswith('.cram'):
            mode = 'rc'
            kwargs = {'reference_filename': reference} if reference else {}
        else:
            mode = 'rb'
            kwargs = {}

        with pysam.AlignmentFile(input_bcram, mode, **kwargs) as samfile:
            chromosomes = list(samfile.references)

        if threads > 1:
            # INFO: Parallel processing by chromosome
            tasks = [
                (input_bcram, chrom, unit_size, tolerance, min_indel_size, reference)
                for chrom in chromosomes
            ]

            with ProcessPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(process_region_wrapper, task): task[1] for task in tasks}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chromosomes", file=sys.stderr):
                    chrom = futures[future]
                    try:
                        indels, read_names = future.result()
                        all_indels.extend(indels)
                        all_read_names.update(read_names)
                    except Exception as e:
                        print(f"Error processing {chrom}: {e}", file=sys.stderr)
        else:
            # INFO: Single-threaded processing
            ref_fasta = None
            if reference:
                try:
                    ref_fasta = pysam.FastaFile(reference)
                except Exception:
                    ref_fasta = None

            try:
                with pysam.AlignmentFile(input_bcram, mode, **kwargs) as samfile:
                    total_reads = None
                    try:
                        total_reads = samfile.mapped + samfile.unmapped
                    except Exception:
                        pass

                    for read in tqdm(samfile.fetch(), total=total_reads, desc="Processing reads", file=sys.stderr):
                        if read.is_unmapped:
                            continue
                        read_indels = parse_cigar_for_indels(read, unit_size, tolerance, min_indel_size, ref_fasta)
                        all_indels.extend(read_indels)
                        if any(ind.is_unit_sized for ind in read_indels):
                            all_read_names.add(read.query_name)
            finally:
                if ref_fasta is not None:
                    ref_fasta.close()

    # INFO: Remove duplicates (same indel from overlapping regions/windows)
    seen = set()
    unique_indels = []
    for indel in all_indels:
        key = (indel.chrom, indel.start, indel.end, indel.indel_type,
               indel.indel_length, indel.read_name)
        if key not in seen:
            seen.add(key)
            unique_indels.append(indel)
    all_indels = unique_indels

    # INFO: Sort by chromosome and position
    all_indels.sort(key=lambda x: (x.chrom, x.start, x.end))

    # INFO: Write output
    if output_bed == "-":
        print(header)
        for indel in all_indels:
            print(indel.to_bed_line())
    else:
        open_func = gzip.open if gzip_output else open
        mode = 'wt' if gzip_output else 'w'

        with open_func(output_bed, mode) as f:
            f.write(header + "\n")
            for indel in all_indels:
                f.write(indel.to_bed_line() + "\n")

    # INFO: Extract reads with unit-sized indels to BAM and/or FASTA
    if reads_output or fasta_output:
        extract_reads(
            input_bcram=input_bcram,
            read_names=all_read_names,
            regions=scan_regions,
            reference=reference,
            bam_output=reads_output,
            fasta_output=fasta_output,
            threads=threads
        )


def main(
    input_bcram: Path,
    *,
    output_bed: str = "-",
    unit_size: int = 171,
    tolerance: float = 0.99,
    min_indel_size: int = 10,
    threads: int = 1,
    total_bed: Optional[Path] = None,
    target_bed: Optional[Path] = None,
    window_size: int = 1000,
    collapse_target: bool = False,
    padding: int = 50000,
    reference: Optional[Path] = None,
    gzip_output: bool = False,
    reads_output: Optional[str] = None,
    fasta_output: Optional[str] = None,
):
    """
    Scan BAM/CRAM files for indels meeting minimum size criteria.

    Parses CIGAR strings to identify insertions and deletions that meet
    the minimum size threshold. Also annotates whether each indel is
    approximately a multiple of a specified unit size (e.g., 171bp for
    alpha-satellite HOR units).

    Output BED columns: chrom, start, end, indel_id, indel_type, indel_length,
    unit_multiple, is_unit_sized, median_bq, overlap_class, gc_identity,
    aligned_fraction, read_length, alignment_span, min_dist_query, min_dist_ref,
    read_name, is_primary, sequence

    Author: Sohny (Min-Hwan Sohn)

    :param input_bcram: Path to the input BAM or CRAM file
    :param output_bed: Path to output BED file (use "-" for stdout)
    :param unit_size: Size of the indel unit to detect (e.g., 171 for alpha-satellite)
    :param tolerance: Tolerance for unit size matching (0.99 = allow 1% deviation)
    :param min_indel_size: Minimum indel size to report (default 10bp)
    :param threads: Number of threads for parallel processing
    :param total_bed: BED file with total intervals containing target regions
    :param target_bed: BED file with target regions (e.g., CDRs) to restrict detection
    :param window_size: Window size for processing within total intervals
    :param collapse_target: Collapse target intervals by contig before processing
    :param padding: Padding in bp to extend target intervals on each side (default 50kb)
    :param reference: Path to reference FASTA (required for CRAM files)
    :param gzip_output: Compress output with gzip
    :param reads_output: Path to output BAM/CRAM with reads containing unit-sized indels
    :param fasta_output: Path to output FASTA with reads containing unit-sized indels
    """

    if not input_bcram.exists():
        print(f"Error: Input file not found: {input_bcram}", file=sys.stderr)
        sys.exit(1)

    if target_bed and not target_bed.exists():
        print(f"Error: Target BED file not found: {target_bed}", file=sys.stderr)
        sys.exit(1)

    if total_bed and not total_bed.exists():
        print(f"Error: Total BED file not found: {total_bed}", file=sys.stderr)
        sys.exit(1)

    if total_bed and not target_bed:
        print("Error: --total-bed requires --target-bed to be specified", file=sys.stderr)
        sys.exit(1)

    ref_path = str(reference) if reference else None
    if str(input_bcram).endswith('.cram') and ref_path is None:
        print("Warning: CRAM file detected but no reference provided. "
              "This may fail if REF_PATH is not set in the file.", file=sys.stderr)

    scan_unitsize_indels(
        input_bcram=str(input_bcram),
        output_bed=output_bed,
        unit_size=unit_size,
        tolerance=tolerance,
        min_indel_size=min_indel_size,
        threads=threads,
        total_bed=str(total_bed) if total_bed else None,
        target_bed=str(target_bed) if target_bed else None,
        window_size=window_size,
        collapse_target=collapse_target,
        padding=padding,
        reference=ref_path,
        gzip_output=gzip_output,
        reads_output=reads_output,
        fasta_output=fasta_output
    )


if __name__ == '__main__':
    defopt.run(main)

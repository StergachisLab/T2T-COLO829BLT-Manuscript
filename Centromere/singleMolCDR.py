#!/usr/bin/env python3
import argparse, gzip, re
import numpy as np
from tqdm import tqdm
import pyfaidx

def parse_regions(path, min_bp=250_000):
    """Return (regions, by_chrom) where
       regions = [(chrom, start, end)]  filtered to > min_bp and sorted
       by_chrom = {chrom: [(start, end, idx), ...]} sorted by start
    """
    regions = []
    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt") as f:
        for line in f:
            if not line.strip() or line[0] == "#": continue
            c, s, e = line.split()[:3]
            s, e = int(s), int(e)
            if e - s > min_bp:
                regions.append((c, s, e))
    #  sort by chrom, then start
    regions = sorted(regions, key=lambda t: (t[0], t[1], t[2]))
    by_chrom = {}
    for idx, (c, s, e) in enumerate(regions):
        by_chrom.setdefault(c, []).append((s, e, idx))
    return regions, by_chrom

def find_cg_regex(seq, substring="CG"):
    return [m.start() for m in re.finditer(substring, seq)]

def generate_read_binary(interval):
    """Return a length-(end-start) 0/1 array with 1s at methylated positions from the last column"""
    start, stop = int(interval[1]), int(interval[2])
    L = stop - start
    if L <= 0:
        return np.zeros(0, dtype=np.int8)
    a = np.zeros(L, dtype=np.int8)

    raw = interval[-1] if len(interval) else ""
    s = str(raw).strip()
    if s and s != ".":
        idx = np.fromstring(s, sep=",", dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < L)]
        if idx.size:
            a[idx] = 1
    return a

def sliding_mcg_content(interval, seq, window_bp=1000, step_bp=1000):
    """
    1 kb windows across the molecule: ratio = mCG / CG 
    Returns dict with starts,ends,proportion

    """
    start, stop = int(interval[1]), int(interval[2])
    L = stop - start
    if L < window_bp:
        return {"starts": np.array([], dtype=np.int64),
                "ends":   np.array([], dtype=np.int64),
                "ratio":  np.array([], dtype=float)}

    strand = interval[5] if len(interval) > 5 else "+"
    motif = "GC" if strand == "-" else "CG"

    ones = generate_read_binary(interval)                         # 0/1 per bp
    motif_starts = np.array(find_cg_regex(seq, motif), dtype=np.int64)
    motif_mask = np.zeros(L, dtype=np.int8)
    if motif_starts.size:
        motif_mask[motif_starts] = 1

    ps_ones  = np.concatenate(([0], np.cumsum(ones)))
    ps_motif = np.concatenate(([0], np.cumsum(motif_mask)))

    starts = np.arange(0, L - window_bp + 1, step_bp, dtype=np.int64)
    ends   = starts + window_bp
    win_ones   = ps_ones[ends]  - ps_ones[starts]
    win_motifs = ps_motif[ends] - ps_motif[starts]

    ratio = np.full_like(win_ones, np.nan, dtype=float)
    nz = win_motifs > 0
    ratio[nz] = win_ones[nz] / win_motifs[nz]

    return {"starts": starts, "ends": ends, "ratio": ratio}

def smooth_islands_binary(labels: np.ndarray, max_len_bins: int = 5) -> np.ndarray:
    """
    Flip short runs (0 or 1) of length <= max_len_bins when flanked by the opposite value
    Edge runs (touching either end) are left unchanged
    """
    a = np.asarray(labels, dtype=np.int8).copy()
    n = a.size
    i = 0
    while i < n:
        j = i
        v = a[i]
        while j < n and a[j] == v:
            j += 1
        run_len = j - i
        if i > 0 and j < n and run_len <= max_len_bins and a[i-1] == a[j] and a[i-1] != v:
            a[i:j] = a[i-1]
        i = j
    return a

def bed12_line(chrom, chromStart, chromEnd, name, strand, blocks,
               score=0, itemRgb="0", thickStart=None, thickEnd=None):
    """blocks : list of (start,end) genomic coords"""
    if thickStart is None: thickStart = chromStart
    if thickEnd   is None: thickEnd   = chromEnd
    blockCount = len(blocks)
    blockSizes  = [max(1, e - s) for s, e in blocks]
    blockStarts = [max(0, s - chromStart) for s, _ in blocks]
    return "\t".join(map(str, [
        chrom, chromStart, chromEnd, name, score, strand,
        thickStart, thickEnd, itemRgb, blockCount,
        ",".join(map(str, blockSizes)) + ",",
        ",".join(map(str, blockStarts)) + ",",
    ]))

# ------------------------- main two-pass pipeline -------------------------

def pass1_region_stats(cpg_path, genome_fa, regions, by_chrom, bin_bp=1000):
    """get per-region mean/std stats from all bins overlapping each region"""
    fa = pyfaidx.Fasta(genome_fa, sequence_always_upper=True)

    n_regions = len(regions)
    sums   = np.zeros(n_regions, dtype=float)
    sums2  = np.zeros(n_regions, dtype=float)
    counts = np.zeros(n_regions, dtype=np.int64)

    open_func = gzip.open if cpg_path.endswith(".gz") else open
    for line in tqdm(open_func(cpg_path, "rt"), desc="Pass 1 (region stats)"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.strip().split()
        if len(f) < 4: continue
        chrom, iv_start, iv_end = f[0], int(f[1]), int(f[2])
        if chrom not in by_chrom:
            continue
        seq = fa[chrom][iv_start:iv_end].seq
        res = sliding_mcg_content(f, seq, window_bp=bin_bp, step_bp=bin_bp)
        if res["ratio"].size == 0:
            continue
        g_starts = iv_start + res["starts"]
        g_ends   = iv_start + res["ends"]
        vals     = res["ratio"]

        # For each region on this chrom, get ratios
        for r_start, r_end, idx in by_chrom[chrom]:
            if r_end <= iv_start or r_start >= iv_end:
                continue
            mask = (g_starts < r_end) & (g_ends > r_start) & np.isfinite(vals)
            if mask.any():
                v = vals[mask]
                sums[idx]  += float(v.sum())
                sums2[idx] += float((v**2).sum())
                counts[idx] += int(v.size)

    # convert to mean,std
    mu  = np.full(n_regions, np.nan, dtype=float)
    sd  = np.full(n_regions, np.nan, dtype=float)
    ok  = counts > 0
    mu[ok] = sums[ok] / counts[ok]
    var = np.zeros_like(sd)
    var[ok] = (sums2[ok] / counts[ok]) - (mu[ok] ** 2)
    var[var < 0] = 0.0
    sd[ok] = np.sqrt(var[ok])
    sd[(~np.isfinite(sd)) | (sd == 0.0)] = np.nan
    return mu, sd

def pass2_call_and_write(cpg_path, genome_fa, regions, by_chrom, mu, sd,
                         out_bed, bin_bp=1000, zthr=-1, island_kb=5, add_caps=True):
    """Apply per-region z to each overlapping molecule, call 0 stretches within regions, write one BED12 file"""
    fa = pyfaidx.Fasta(genome_fa, sequence_always_upper=True)
    out = open(out_bed, "w")

    open_func = gzip.open if cpg_path.endswith(".gz") else open
    for line in tqdm(open_func(cpg_path, "rt"), desc="Pass 2 (calling + BED12)"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.strip().split()
        if len(f) < 4: continue
        chrom, iv_start, iv_end = f[0], int(f[1]), int(f[2])
        name   = str(f[3])
        strand = f[5] if len(f) > 5 else "."

        blocks = []  

        if chrom in by_chrom:
            seq = fa[chrom][iv_start:iv_end].seq
            res = sliding_mcg_content(f, seq, window_bp=bin_bp, step_bp=bin_bp)
            if res["ratio"].size:
                g_starts = iv_start + res["starts"]
                g_ends   = iv_start + res["ends"]
                vals     = res["ratio"]

                # Bin length for island smoothing
                bin_len_bp = int(np.median(g_ends - g_starts)) if g_starts.size else bin_bp
                if bin_len_bp <= 0: bin_len_bp = bin_bp
                max_island_bins = max(1, int(round((island_kb * 1000) / bin_len_bp)))

                # Iterate regions on this chrom that overlap the molecule
                for r_start, r_end, idx in by_chrom[chrom]:
                    if r_end <= iv_start or r_start >= iv_end:
                        continue
                    if not np.isfinite(mu[idx]) or not np.isfinite(sd[idx]):
                        continue  # no usable stats for this region

                    # Windows overlapping this region
                    mask = (g_starts < r_end) & (g_ends > r_start)
                    if not mask.any():
                        continue

                    v = vals[mask]
                    gs = g_starts[mask]
                    ge = g_ends[mask]

                    # Apply per-region z
                    z = (v - mu[idx]) / sd[idx] if sd[idx] > 0 else np.zeros_like(v)

                    # Threshold 0/1 then smooth within-region only
                    lab = np.ones_like(z, dtype=np.int8)
                    fin = np.isfinite(z)
                    lab[fin & (z <= zthr)] = 0
                    lab = smooth_islands_binary(lab, max_len_bins=max_island_bins)

                    # Emit 0 runs clipped to region bounsd
                    i = 0
                    n = lab.size
                    while i < n:
                        if lab[i] != 0:
                            i += 1
                            continue
                        j = i
                        while j < n and lab[j] == 0:
                            j += 1
                        seg_s = max(int(gs[i]), r_start)
                        seg_e = min(int(ge[j-1]), r_end)
                        if seg_e > seg_s:
                            blocks.append((seg_s, seg_e))
                        i = j

        #  include 1-bp caps at molecule boundaries if requested
        if add_caps:
            blocks.append((iv_start, iv_start + 1))
            blocks.append((iv_end - 1, iv_end))

        # Merge amd sort
        if blocks:
            blocks.sort()
            merged = []
            for s, e in blocks:
                if not merged or s > merged[-1][1]:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            blocks = [(s, e) for s, e in merged]

        rec = bed12_line(chrom, iv_start, iv_end, name, strand, blocks)
        out.write(rec + "\n")

    out.close()

def main():
    ap = argparse.ArgumentParser(
        description="Per-region z-score calling on molecules; emit one BED12 (with .bed extension)."
    )
    ap.add_argument("--regions", required=True, help="Regions BED (used to compute per-region z; only regions > 250kb are kept)")
    ap.add_argument("--cpg",     required=True, help="Molecule file: chrom start end name ... strand ... methyl_positions")
    ap.add_argument("--genome",  required=True, help="Reference FASTA (indexed with .fai)")
    ap.add_argument("-o", "--output", default="molecule_calls.bed", help="Output BED12 file (default: molecule_calls.bed)")
    ap.add_argument("--bin", type=int, default=1000, help="Bin size (bp), default 1000")
    ap.add_argument("--zthr", type=float, default=-1, help="Threshold for label 0 on per-region z (default: -1.5)")
    ap.add_argument("--island_kb", type=int, default=5, help="Remove islands ≤ this size (kb) (default: 5)")
    ap.add_argument("--min_region_bp", type=int, default=250_000, help="Keep only regions longer than this (default: 250000)")
    ap.add_argument("--no_caps", action="store_true", help="Do not add 1-bp caps at molecule start/end")
    args = ap.parse_args()

    regions, by_chrom = parse_regions(args.regions, min_bp=args.min_region_bp)
    if not regions:
        raise SystemExit("No regions ≥ min_region_bp found in --regions.")

    mu, sd = pass1_region_stats(args.cpg, args.genome, regions, by_chrom, bin_bp=args.bin)
    pass2_call_and_write(
        args.cpg, args.genome, regions, by_chrom, mu, sd,
        out_bed=args.output, bin_bp=args.bin, zthr=args.zthr,
        island_kb=args.island_kb, add_caps=not args.no_caps
    )

if __name__ == "__main__":
    main()

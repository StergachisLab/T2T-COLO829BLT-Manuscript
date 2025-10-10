#!/usr/bin/env python3
import argparse, os, sys, tempfile, subprocess
from typing import Dict, List, Tuple
import numpy as np
import pyBigWig
from tqdm import tqdm

BIN = 1000  # 1 kb


#I/O helpers 
def read_bed(path: str, min_region_bp: int = 100_000) :
    d = {}
    with (open(path, "rt") if not path.endswith(".gz") else __import__("gzip").open(path, "rt")) as f:
        for line in f:
            if not line or line[0] in "#tb":  # skip comments/track/browser
                if line.startswith("track") or line.startswith("browser") or line.startswith("#"):
                    continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            c, s, e = parts[0], int(parts[1]), int(parts[2])
            if e > s and (e - s) > min_region_bp:
                d.setdefault(c, []).append((s, e))
    for c in d:
        d[c].sort()
    return d

def write_bed(recs: List[Tuple[str,int,int]], out_bed: str):
    with open(out_bed, "w") as out:
        for chrom, start, end in recs:
            if end - start > 5_000:
                out.write(f"{chrom}\t{start}\t{end}\n")

# core code 

def bw_bin_means(bw: pyBigWig.pyBigWig, chrom: str, start: int, end: int, bin_size=BIN):
    """Return (bin_starts, means) for full 1kb bins inside [start,end)."""
    n_bins = (end - start) // bin_size
    if n_bins <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    stop = start + n_bins * bin_size
    vals = bw.stats(chrom, start, stop, nBins=n_bins, type="mean")
    means = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    bin_starts = start + np.arange(n_bins, dtype=np.int64) * bin_size
    return bin_starts, means

def per_chrom_distribution(bed: Dict[str, List[Tuple[int,int]]], bw_path: str, bin_size=BIN):
    """Pass 1: collect all bin means per chromosome for z-score statistics (only from filtered BED)"""
    stats = {}  # chrom -> (mean, std)
    with pyBigWig.open(bw_path) as bw:
        for chrom, regions in bed.items():
            all_vals = []
            for s, e in regions:
                _, means = bw_bin_means(bw, chrom, s, e, bin_size)
                if means.size:
                    all_vals.append(means)
            if all_vals:
                v = np.concatenate(all_vals)
                v = v[np.isfinite(v)]
                if v.size:
                    mu = float(v.mean())
                    sd = float(v.std(ddof=0))
                else:
                    mu, sd = np.nan, np.nan
            else:
                mu, sd = np.nan, np.nan
            stats[chrom] = (mu, sd if sd > 0 else np.nan)
    return stats

def smooth_islands(labels: np.ndarray, max_len_bins: int):
    """Flip runs (0 or 1) of length <= max_len_bins when flanked by the opposite value"""
    a = labels.copy()
    n = a.size
    i = 0
    while i < n:
        j = i
        v = a[i]
        while j < n and a[j] == v:
            j += 1
        run_len = j - i
        left_ok  = (i > 0)
        right_ok = (j < n)
        if left_ok and right_ok and run_len <= max_len_bins and a[i-1] != v and a[j] == a[i-1]:
            a[i:j] = a[i-1]
        i = j
    return a

def label_and_islands_to_bed(
    bed: Dict[str, List[Tuple[int,int]]],
    bw_path: str,
    bin_size=BIN,
    zthr=-1.5,
    max_island_kb=5,
    edge_buffer_bp=50_000
) :
    """
    per-region z-scores:
    mean stdare computed from the 1kb means inside each BED interval separately
    label 0 if z < zthr else 1
    smooth islands (runs ≤ max_island_kb) if flanked by opposite value
    emit only 0 regions, clipped ≥ edge_buffer_bp from both ends
    """
    out = []
    max_island_bins = max(1, int(np.floor((max_island_kb * 1000) / bin_size)))

    with pyBigWig.open(bw_path) as bw:
        for chrom, regions in tqdm(bed.items()):
            for start, end in regions:
                bin_starts, means = bw_bin_means(bw, chrom, start, end, bin_size)
                if means.size == 0:
                    continue

                # per-region mean and std
                finite = np.isfinite(means)
                if not np.any(finite):
                    continue  # nothing usable in this region

                mu = float(np.mean(means[finite]))
                sd = float(np.std(means[finite], ddof=0))

                if sd == 0.0 or not np.isfinite(sd):
                    z = np.zeros_like(means)
                else:
                    z = (means - mu) / sd

                labels = np.where(z < zthr, 0, 1).astype(np.int8)
                labels = smooth_islands(labels, max_len_bins=max_island_bins)

                interior_start = start + edge_buffer_bp
                interior_end   = end   - edge_buffer_bp
                if interior_end <= interior_start:
                    continue

                i, n = 0, labels.size
                while i < n:
                    if labels[i] != 0:
                        i += 1
                        continue
                    j = i
                    while j < n and labels[j] == 0:
                        j += 1
                    seg_start = int(bin_starts[i])
                    seg_end   = int(bin_starts[j-1] + bin_size)

                    seg_start = max(seg_start, interior_start)
                    seg_end   = min(seg_end,   interior_end)
                    if seg_end > seg_start:
                        out.append((chrom, seg_start, seg_end))
                    i = j

    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return out


def merge_bed_runs(bed3: List[Tuple[str,int,int]]):
    """Merge overlapping or adjacent 0runs"""
    if not bed3:
        return bed3
    merged = []
    cur_c, cur_s, cur_e = bed3[0]
    for c, s, e in bed3[1:]:
        if c == cur_c and s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_c, cur_s, cur_e))
            cur_c, cur_s, cur_e = c, s, e
    merged.append((cur_c, cur_s, cur_e))
    return merged


def main():
    p = argparse.ArgumentParser(
        description="Call low-methylation islands by z-score from BigWig over BED intervals (>100kb), "
                    "smooth short islands, enforce >= 50kb from interval ends, output 0 regions as BED."
    )
    p.add_argument("bed", help="BED of regions of interest")
    p.add_argument("bigwig", help="BigWig with CpG methylation rate (0 to 1)")
    p.add_argument("-o", "--output", default="lowMethIslands", help="Output prefix (produces .bed)")
    p.add_argument("--zthr", type=float, default=-1.5, help="Z-score threshold for label 0 (default: -1.5)")
    p.add_argument("--bin",  type=int,   default=BIN, help="Bin size (bp), default 1000")
    p.add_argument("--island_kb", type=int, default=5, help="Max island length to smooth (kb) (default: 5)")
    p.add_argument("--min_region_bp", type=int, default=100_000, help="Only consider BED regions longer than this")
    p.add_argument("--edge_buffer_bp", type=int, default=50_000, help="Clip outputs to be this far from interval ends")
    args = p.parse_args()

    bed_dict = read_bed(args.bed, min_region_bp=args.min_region_bp)

    stats = None

    bed0 = label_and_islands_to_bed(
        bed_dict, args.bigwig, stats,
        bin_size=args.bin, zthr=args.zthr,
        max_island_kb=args.island_kb, edge_buffer_bp=args.edge_buffer_bp)

    bed0 = merge_bed_runs(bed0)

    out_bed = args.output + ".bed"
    write_bed(bed0, out_bed)
    sys.stderr.write(f"[info] Wrote BED with {len(bed0)} low-methylation regions (≥{args.edge_buffer_bp} bp from ends): {out_bed}\n")

if __name__ == "__main__":
    main()


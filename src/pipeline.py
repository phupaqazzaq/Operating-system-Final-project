#!/usr/bin/env python3
"""
pipeline.py — Main OS-Optimized Pipeline for Thai Harassment Detection.

Combines: Memory mgmt, Process mgmt, Synchronization, I/O, Scheduling, File mgmt.
Run: python pipeline.py [--all-benchmarks]
"""
import os, sys, time, json, csv
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_with_mmap, load_with_read, benchmark_loaders
from preprocessor import preprocess_chunk, split_chunks, load_csv, benchmark_workers
from sync_queue import benchmark_buffer_sizes
from io_benchmark import benchmark_io
from scheduler_demo import benchmark_scheduling
from file_manager import benchmark_file_management, atomic_write

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'thai_toxicity.csv')

def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Thai Harassment Detection — OS-Optimized Pipeline        ║")
    print("║   Dataset: thai_toxicity_2025_train_final (22,855 samples) ║")
    print("║   Task: Binary toxic/non-toxic classification              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

def run_pipeline():
    # ── STEP 1: Memory Management ────────────────────────────────
    print("━" * 65)
    print("STEP 1: Data Loading (Memory Management — mmap vs read)")
    print("━" * 65)
    rows_mm, t_mm, sz = load_with_mmap(DATA)
    rows_rd, t_rd, _  = load_with_read(DATA)
    print(f"  mmap(): {t_mm:.4f}s | read(): {t_rd:.4f}s | {len(rows_mm):,} rows | {sz/1024:.1f} KB")
    data = rows_mm if t_mm < t_rd else rows_rd
    winner = "mmap()" if t_mm < t_rd else "read()"
    print(f"  → Using {winner}\n")

    # ── STEP 2: Multiprocessing ──────────────────────────────────
    print("━" * 65)
    print("STEP 2: Preprocessing (fork + multiprocessing.Pool)")
    print("━" * 65)
    n_w = min(os.cpu_count() or 2, 4)
    try:
        cpus = os.sched_getaffinity(0)
        print(f"  Available CPUs: {cpus}")
    except AttributeError:
        pass

    chunks = split_chunks(data, n_w)
    args = [(c, i) for i, c in enumerate(chunks)]
    start = time.perf_counter()
    with mp.Pool(n_w) as pool:
        all_res = pool.map(preprocess_chunk, args)
    t_pre = time.perf_counter() - start
    flat = [item for batch in all_res for item in batch]

    toxic = sum(1 for r in flat if r['label'] == 1)
    avg_tok = sum(r['token_count'] for r in flat) / len(flat) if flat else 0
    print(f"  {n_w} workers | {len(flat):,} samples | {t_pre:.4f}s | {len(flat)/t_pre:.0f} samples/sec")
    print(f"  Toxic: {toxic:,} | Non-toxic: {len(flat)-toxic:,} | Avg tokens: {avg_tok:.1f}\n")

    # ── STEP 3: Save (I/O + File Management) ─────────────────────
    print("━" * 65)
    print("STEP 3: Save Results (atomic write — temp → fsync → rename)")
    print("━" * 65)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(out_dir, 'thai_preprocessed.csv')

    start = time.perf_counter()
    lines = ['text,tokens,token_count,label,pid']
    for r in flat:
        tok_json = json.dumps(r['tokens'], ensure_ascii=False).replace(',', ';')
        lines.append(f"{r['text']},{tok_json},{r['token_count']},{r['label']},{r['pid']}")
    atomic_write(out_path, '\n'.join(lines))
    t_save = time.perf_counter() - start
    fsize = os.path.getsize(out_path)
    print(f"  Output: {out_path}")
    print(f"  Size: {fsize/1024:.1f} KB | Time: {t_save:.4f}s\n")

    # ── Summary ──────────────────────────────────────────────────
    total = t_mm + t_pre + t_save
    print("━" * 65)
    print("PIPELINE SUMMARY")
    print("━" * 65)
    print(f"  Samples:       {len(flat):,}")
    print(f"  Data loading:  {t_mm:.4f}s  (mmap)")
    print(f"  Preprocessing: {t_pre:.4f}s  ({n_w} workers)")
    print(f"  Saving:        {t_save:.4f}s  (atomic write)")
    print(f"  ────────────────────────")
    print(f"  Total:         {total:.4f}s")
    print(f"  Throughput:    {len(flat)/total:,.0f} samples/sec")
    print()
    print("  OS Components:")
    print("    ✓ mmap()              — Memory-mapped loading")
    print("    ✓ fork() / Pool       — Multiprocessing")
    print("    ✓ Lock / Semaphore    — Synchronization")
    print("    ✓ sched_getaffinity() — CPU scheduling")
    print("    ✓ fsync() + rename()  — Atomic file writes")
    print("    ✓ stat() / unlink()   — File management")
    print("━" * 65)

def run_all_benchmarks():
    print("\n" + "█" * 65)
    print("  ALL BENCHMARKS (for Performance Trade-offs section)")
    print("█" * 65 + "\n")
    benchmark_loaders(DATA)
    print()
    benchmark_workers(load_csv(DATA))
    print()
    benchmark_buffer_sizes(DATA)
    print()
    benchmark_io()
    print()
    benchmark_scheduling()
    print()
    benchmark_file_management()

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    print_header()
    if not os.path.exists(DATA):
        print(f"[ERROR] {DATA} not found. Place your CSV in data/thai_toxicity.csv")
        sys.exit(1)
    run_pipeline()
    if '--all-benchmarks' in sys.argv:
        run_all_benchmarks()
    else:
        print("\n[TIP] Run with --all-benchmarks for full performance analysis")

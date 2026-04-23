#!/usr/bin/env python3
"""
preprocessor.py — Process Management & Synchronization.

System calls: fork() (via Pool), wait(), getpid(), futex() (via Lock)

Trade-off: More workers = faster UP TO CPU count. Beyond that, context switch overhead dominates.
Each forked worker uses ~50-100MB for the Python interpreter.
"""
import os, sys, time, csv, re, json
import multiprocessing as mp
from multiprocessing import Value, Lock

class SharedProgress:
    """Thread-safe counter using Lock (mutex). Uses futex() syscall internally."""
    def __init__(self):
        self.count = Value('i', 0)
        self.lock = Lock()
    def increment(self, n=1):
        with self.lock:
            self.count.value += n
    def get(self):
        with self.lock:
            return self.count.value

def preprocess_chunk(args):
    """Preprocess Thai text in a forked child process."""
    chunk, chunk_id = args
    pid = os.getpid()
    try:
        from pythainlp.tokenize import word_tokenize
        has_pythai = True
    except ImportError:
        has_pythai = False

    results = []
    for row in chunk:
        text = row.get('text', '')
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text, engine='newmm') if has_pythai and text else text.split()
        try:
            label = float(row.get('label', 0))
        except (ValueError, TypeError):
            label = 0.0
        results.append({
            'text': text, 'tokens': tokens, 'token_count': len(tokens),
            'label': int(label), 'pid': pid
        })
    return results

def split_chunks(data, n):
    sz = max(1, len(data) // n)
    return [data[i:i+sz] for i in range(0, len(data), sz)]

def load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def benchmark_workers(data, max_w=None):
    """Compare 1 vs 2 vs 4 vs N workers."""
    if max_w is None:
        max_w = min(os.cpu_count() or 2, 8)
    print("=" * 65)
    print("BENCHMARK: Process Management — Multiprocessing Workers")
    print(f"  Dataset: {len(data):,} samples | CPUs: {os.cpu_count()} | Parent PID: {os.getpid()}")
    print("=" * 65)

    counts = sorted(set(w for w in [1, 2, 4, max_w] if w <= max_w))
    results = {}
    for n in counts:
        chunks = split_chunks(data, n)
        args = [(c, i) for i, c in enumerate(chunks)]
        start = time.perf_counter()
        if n == 1:
            all_res = [preprocess_chunk(a) for a in args]
        else:
            with mp.Pool(n) as pool:
                all_res = pool.map(preprocess_chunk, args)
        elapsed = time.perf_counter() - start
        total = sum(len(r) for r in all_res)
        pids = set(item['pid'] for batch in all_res for item in batch)
        results[n] = elapsed
        print(f"  {n} worker(s): {elapsed:.4f}s | {total:,} samples | PIDs: {pids}")

    print("\n  --- Speedup Analysis ---")
    base = results[1]
    for n, t in results.items():
        sp = base / t if t > 0 else 0
        eff = sp / n * 100
        print(f"  {n} workers: {t:.4f}s | speedup={sp:.2f}x | efficiency={eff:.0f}%")
    print("=" * 65)
    return results

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    fp = os.path.join(os.path.dirname(__file__), '..', 'data', 'thai_toxicity.csv')
    if not os.path.exists(fp):
        print("[ERROR] Dataset not found"); sys.exit(1)
    data = load_csv(fp)
    if '--benchmark' in sys.argv:
        benchmark_workers(data)
    else:
        n = min(os.cpu_count() or 2, 4)
        chunks = split_chunks(data, n)
        args = [(c, i) for i, c in enumerate(chunks)]
        start = time.perf_counter()
        with mp.Pool(n) as pool:
            all_res = pool.map(preprocess_chunk, args)
        flat = [item for batch in all_res for item in batch]
        print(f"[DONE] Preprocessed {len(flat):,} samples in {time.perf_counter()-start:.3f}s with {n} workers")

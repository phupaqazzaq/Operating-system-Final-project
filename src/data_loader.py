#!/usr/bin/env python3
"""
data_loader.py — Memory Management: mmap() vs read() for dataset loading.

System calls: open(), fstat(), mmap(), munmap(), read(), close()

Trade-off:
  - mmap(): zero-copy, lazy loading via page faults. Best for large/random access.
  - read(): copies kernel→user buffer. Simpler, better for small sequential reads.
"""
import mmap, os, time, csv, io

def load_with_mmap(filepath):
    """Load CSV via mmap() — file mapped directly into process virtual memory."""
    start = time.perf_counter()
    fd = os.open(filepath, os.O_RDONLY)
    file_size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
    raw = mm.read()
    text_data = raw.decode('utf-8-sig')  # handle BOM
    reader = csv.DictReader(io.StringIO(text_data))
    rows = list(reader)
    mm.close()
    os.close(fd)
    return rows, time.perf_counter() - start, file_size

def load_with_read(filepath):
    """Load CSV via standard buffered read()."""
    start = time.perf_counter()
    file_size = os.path.getsize(filepath)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return rows, time.perf_counter() - start, file_size

def load_with_chunks(filepath, chunk_size=8192):
    """Load via explicit chunked os.read() — demonstrates buffer size trade-off."""
    start = time.perf_counter()
    fd = os.open(filepath, os.O_RDONLY)
    raw = bytearray()
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        raw.extend(chunk)
    os.close(fd)
    text_data = raw.decode('utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(text_data)))
    return rows, time.perf_counter() - start

def benchmark_loaders(filepath):
    """Benchmark all loading strategies."""
    print("=" * 65)
    print("BENCHMARK: Memory Management — mmap() vs read()")
    print("=" * 65)
    results = {}

    rows, t, sz = load_with_mmap(filepath)
    results['mmap()'] = t
    print(f"  mmap()           : {t:.4f}s | {len(rows):,} rows | {sz/1024:.1f} KB")

    rows, t, _ = load_with_read(filepath)
    results['read() buffered'] = t
    print(f"  read() buffered  : {t:.4f}s | {len(rows):,} rows")

    rows, t = load_with_chunks(filepath, 512)
    results['read() 512B'] = t
    print(f"  read() 512B      : {t:.4f}s | {len(rows):,} rows")

    rows, t = load_with_chunks(filepath, 65536)
    results['read() 64KB'] = t
    print(f"  read() 64KB      : {t:.4f}s | {len(rows):,} rows")

    fastest = min(results, key=results.get)
    slowest = max(results, key=results.get)
    print(f"\n  Fastest: {fastest} ({results[fastest]:.4f}s)")
    print(f"  Slowest: {slowest} ({results[slowest]:.4f}s)")
    print(f"  Speedup: {results[slowest]/results[fastest]:.2f}x")
    print("=" * 65)
    return results

if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'data', 'thai_toxicity.csv')
    if not os.path.exists(fp):
        print(f"[ERROR] {fp} not found"); sys.exit(1)
    benchmark_loaders(fp)

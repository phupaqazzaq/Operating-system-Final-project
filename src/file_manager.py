#!/usr/bin/env python3
"""
file_manager.py — File Management: Caching, atomic writes, metadata.
Syscalls: stat(), rename(), mkstemp(), fsync(), unlink()
"""
import os, time, tempfile

def atomic_write(filepath, content):
    """Atomic write: temp file → fsync → rename. Crash-safe."""
    d = os.path.dirname(filepath) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        os.write(fd, content.encode('utf-8'))
        os.fsync(fd)
        os.close(fd)
        os.rename(tmp, filepath)  # atomic on same filesystem
    except Exception:
        os.close(fd)
        os.unlink(tmp)
        raise

def benchmark_cache(filepath):
    """Cold vs warm read — demonstrates OS page cache."""
    sz = os.path.getsize(filepath)
    times = []
    for label in ['Cold', 'Warm 1', 'Warm 2']:
        start = time.perf_counter()
        with open(filepath, 'rb') as f:
            _ = f.read()
        times.append((label, time.perf_counter() - start))
    return sz, times

def benchmark_file_management():
    print("=" * 65)
    print("BENCHMARK: File Management — Cache, Atomic Write, Metadata")
    print("=" * 65)
    d = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(d, 'thai_toxicity.csv')

    if os.path.exists(csv_path):
        print("\n  --- Page Cache (cold vs warm read) ---")
        sz, times = benchmark_cache(csv_path)
        for label, t in times:
            print(f"  {label:>6}: {t:.4f}s")
        if times[1][1] > 0:
            print(f"  Cache speedup: {times[0][1]/times[1][1]:.2f}x")

    print("\n  --- Atomic vs Direct Write ---")
    content = "ข้อความทดสอบ\n" * 10000
    fp1 = os.path.join(d, '_direct.txt')
    start = time.perf_counter()
    with open(fp1, 'w') as f: f.write(content)
    t_direct = time.perf_counter() - start

    fp2 = os.path.join(d, '_atomic.txt')
    start = time.perf_counter()
    atomic_write(fp2, content)
    t_atomic = time.perf_counter() - start

    print(f"  Direct write: {t_direct:.4f}s")
    print(f"  Atomic write: {t_atomic:.4f}s (temp→fsync→rename)")
    for f in [fp1, fp2]:
        if os.path.exists(f): os.unlink(f)

    print("\n  --- stat() syscall speed ---")
    if os.path.exists(csv_path):
        start = time.perf_counter()
        for _ in range(10000):
            os.stat(csv_path)
        t = time.perf_counter() - start
        print(f"  10,000 × stat(): {t:.4f}s ({t/10000*1e6:.1f} µs/call)")
    print("=" * 65)

if __name__ == '__main__':
    benchmark_file_management()

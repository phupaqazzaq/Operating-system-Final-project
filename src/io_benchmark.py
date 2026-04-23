#!/usr/bin/env python3
"""
io_benchmark.py — I/O Management: Write strategies comparison.

Syscalls: open(), write(), fsync(), close(), unlink()

Trade-off: Buffered = fast but risk data loss on crash.
           fsync per row = durable but 100x slower.
"""
import os, time, csv, json

def generate_thai_data(n=5000):
    return [{'text': f'ข้อความทดสอบหมายเลข {i} สำหรับการทดสอบ', 'tokens': '["ข้อความ","ทดสอบ"]',
             'label': i % 2, 'count': 4} for i in range(n)]

def write_buffered(data, fp):
    start = time.perf_counter()
    with open(fp, 'w', encoding='utf-8', buffering=8192) as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    return time.perf_counter() - start

def write_unbuffered(data, fp):
    start = time.perf_counter()
    fd = os.open(fp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, (','.join(data[0].keys()) + '\n').encode())
    for row in data:
        os.write(fd, (','.join(str(v) for v in row.values()) + '\n').encode())
    os.close(fd)
    return time.perf_counter() - start

def write_fsync_every(data, fp, n=500):
    """fsync after every row — maximum durability, very slow."""
    subset = data[:n]
    start = time.perf_counter()
    fd = os.open(fp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, (','.join(subset[0].keys()) + '\n').encode())
    for row in subset:
        os.write(fd, (','.join(str(v) for v in row.values()) + '\n').encode())
        os.fsync(fd)
    os.close(fd)
    return time.perf_counter() - start, len(subset)

def write_fsync_batched(data, fp, batch=100):
    start = time.perf_counter()
    fd = os.open(fp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, (','.join(data[0].keys()) + '\n').encode())
    for i, row in enumerate(data):
        os.write(fd, (','.join(str(v) for v in row.values()) + '\n').encode())
        if (i + 1) % batch == 0:
            os.fsync(fd)
    os.fsync(fd)
    os.close(fd)
    return time.perf_counter() - start

def benchmark_io():
    print("=" * 65)
    print("BENCHMARK: I/O Management — Write Strategies")
    print("=" * 65)
    data = generate_thai_data(5000)
    d = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(d, exist_ok=True)
    results = {}

    fp = os.path.join(d, '_b1.csv')
    t = write_buffered(data, fp)
    results['Buffered 8KB'] = t
    print(f"  Buffered (8KB)        : {t:.4f}s | {len(data):,} rows | {os.path.getsize(fp)/1024:.1f}KB")
    os.unlink(fp)

    fp = os.path.join(d, '_b2.csv')
    t = write_unbuffered(data, fp)
    results['Unbuffered'] = t
    print(f"  Unbuffered (per-row)  : {t:.4f}s | {len(data):,} rows")
    os.unlink(fp)

    fp = os.path.join(d, '_b3.csv')
    t, n = write_fsync_every(data, fp)
    results['fsync/row'] = t
    print(f"  fsync() per row       : {t:.4f}s | {n} rows | {n/t:.0f} rows/s")
    os.unlink(fp)

    fp = os.path.join(d, '_b4.csv')
    t = write_fsync_batched(data, fp, 100)
    results['fsync/100'] = t
    print(f"  fsync() per 100 rows  : {t:.4f}s | {len(data):,} rows")
    os.unlink(fp)

    fp = os.path.join(d, '_b5.csv')
    t = write_fsync_batched(data, fp, 1000)
    results['fsync/1000'] = t
    print(f"  fsync() per 1000 rows : {t:.4f}s | {len(data):,} rows")
    os.unlink(fp)

    print("\n  --- Trade-off Summary ---")
    print("  Buffered   → fastest, risk losing data on crash")
    print("  fsync/1000 → good balance of speed + durability")
    print("  fsync/row  → safest but ~100x slower (disk latency per call)")
    print("=" * 65)
    return results

if __name__ == '__main__':
    benchmark_io()

#!/usr/bin/env python3
"""
scheduler_demo.py — CPU Scheduling: nice values and CPU affinity.
Syscalls: nice(), sched_setaffinity(), sched_getaffinity(), getpid()
"""
import os, time, multiprocessing as mp

def cpu_work(n=2_000_000):
    r = 0
    for i in range(n):
        r += i * i % 1000003
    return r

def worker_affinity(cpu_set, n_iter, wid):
    pid = os.getpid()
    try:
        os.sched_setaffinity(0, cpu_set)
        aff = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        aff = "n/a"
    start = time.perf_counter()
    cpu_work(n_iter)
    return {'wid': wid, 'pid': pid, 'affinity': str(aff), 'time': time.perf_counter() - start}

def benchmark_scheduling():
    print("=" * 65)
    print("BENCHMARK: CPU Scheduling — Affinity & Context Switches")
    print(f"  CPUs: {os.cpu_count()} | PID: {os.getpid()}")
    print("=" * 65)
    n_iter = 1_000_000
    cpus = os.cpu_count() or 2

    print("\n  --- CPU Affinity: pinned vs unpinned ---")
    # Unpinned
    start = time.perf_counter()
    for i in range(4):
        r = worker_affinity(set(range(cpus)), n_iter, i)
    t_unpin = time.perf_counter() - start
    print(f"  Unpinned (all cores): {t_unpin:.4f}s")

    # Pinned
    start = time.perf_counter()
    for i in range(min(4, cpus)):
        r = worker_affinity({i % cpus}, n_iter, i)
    t_pin = time.perf_counter() - start
    print(f"  Pinned (1 core each): {t_pin:.4f}s")

    print("\n  --- Context Switch Overhead ---")
    for np in [1, 4, 16]:
        start = time.perf_counter()
        procs = [mp.Process(target=cpu_work, args=(n_iter // np,)) for _ in range(np)]
        for p in procs: p.start()
        for p in procs: p.join()
        print(f"  {np:>2} processes: {time.perf_counter()-start:.4f}s")

    print("\n  Analysis: CPU pinning keeps L1/L2 cache warm.")
    print("  Too many processes → context switch overhead dominates.")
    print("=" * 65)

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    benchmark_scheduling()

#!/usr/bin/env python3
"""
sync_queue.py — Synchronization: Producer-Consumer Bounded Buffer.

Primitives: Mutex (Lock), Semaphore (counting), Condition Variable
Syscalls: futex() (Linux), pthread_mutex_lock/unlock, sem_wait/post

Classic OS problem: Producer reads CSV rows → shared buffer → Consumer preprocesses.
Trade-off: Small buffer → more blocking. Large buffer → more memory, less blocking.
"""
import threading, time, os, sys, csv, re

class BoundedBuffer:
    """Bounded buffer with mutex + semaphores (classic OS synchronization)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.mutex = threading.Lock()
        self.empty_slots = threading.Semaphore(capacity)
        self.full_slots = threading.Semaphore(0)
        self.produced = 0
        self.consumed = 0

    def produce(self, item):
        self.empty_slots.acquire()    # sem_wait: block if full
        with self.mutex:              # pthread_mutex_lock
            self.buffer.append(item)
            self.produced += 1
        self.full_slots.release()     # sem_post: signal consumer

    def consume(self):
        self.full_slots.acquire()     # sem_wait: block if empty
        with self.mutex:              # pthread_mutex_lock
            item = self.buffer.pop(0)
            self.consumed += 1
        self.empty_slots.release()    # sem_post: signal producer
        return item

SENTINEL = "###DONE###"

def run_pipeline(filepath, buffer_size=100, n_consumers=2):
    buf = BoundedBuffer(buffer_size)
    output = []
    output_lock = threading.Lock()

    def producer():
        count = 0
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                buf.produce(row)
                count += 1
        for _ in range(n_consumers):
            buf.produce(SENTINEL)
        print(f"    Producer: {count:,} items")

    def consumer(cid):
        count = 0
        while True:
            item = buf.consume()
            if item == SENTINEL:
                break
            text = re.sub(r'\s+', ' ', item.get('text', '')).strip()
            tokens = text.split()
            with output_lock:
                output.append({'text': text, 'token_count': len(tokens), 'label': item.get('label', 0)})
                count += 1
        print(f"    Consumer-{cid}: {count:,} items")

    start = time.perf_counter()
    prod = threading.Thread(target=producer)
    cons = [threading.Thread(target=consumer, args=(i,)) for i in range(n_consumers)]
    prod.start()
    for c in cons: c.start()
    prod.join()
    for c in cons: c.join()
    elapsed = time.perf_counter() - start
    return elapsed, len(output), buf

def benchmark_buffer_sizes(filepath):
    print("=" * 65)
    print("BENCHMARK: Synchronization — Bounded Buffer (Producer-Consumer)")
    print("=" * 65)
    for bs in [10, 50, 100, 500, 1000]:
        elapsed, count, buf = run_pipeline(filepath, buffer_size=bs, n_consumers=2)
        print(f"  Buffer={bs:>5}: {elapsed:.4f}s | {count:,} items | produced={buf.produced:,} consumed={buf.consumed:,}\n")

    print("  --- Analysis ---")
    print("  Smaller buffer → more semaphore waits (blocking) → slower")
    print("  Larger buffer → fewer blocks, but higher memory usage")
    print("  The Lock (mutex) serializes access — this is the bottleneck")
    print("=" * 65)

if __name__ == '__main__':
    fp = os.path.join(os.path.dirname(__file__), '..', 'data', 'thai_toxicity.csv')
    if not os.path.exists(fp):
        print("[ERROR] Dataset not found"); sys.exit(1)
    benchmark_buffer_sizes(fp)

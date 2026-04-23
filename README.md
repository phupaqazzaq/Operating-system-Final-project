# Thai Harassment Detection — OS-Optimized Pipeline

## Project Overview
Thai text harassment detection system with explicit OS-level optimizations
demonstrating CPU scheduling, memory management, multiprocessing, synchronization,
and file I/O management.

**Dataset:** `malexandersalazar/toxicity-multilingual-binary-classification-dataset` (2025)  
- Filtered for Thai language → 22,855 samples (8,429 toxic / 14,426 non-toxic)  
- Binary classification: toxic (1) / non-toxic (0)  
- File: `data/thai_toxicity.csv` (7.04 MB)

## OS Components Implemented

| OS Area               | File                  | Key System Calls                           |
|-----------------------|-----------------------|--------------------------------------------|
| Memory Management     | `data_loader.py`      | `mmap()`, `munmap()`, `fstat()`, `read()`  |
| Process Management    | `preprocessor.py`     | `fork()`, `wait()`, `getpid()`             |
| Synchronization       | `sync_queue.py`       | `Lock` (mutex), `Semaphore`, `Condition`   |
| CPU Scheduling        | `scheduler_demo.py`   | `sched_setaffinity()`, `nice()`            |
| I/O Management        | `io_benchmark.py`     | `open()`, `write()`, `fsync()`, `unlink()` |
| File Management       | `file_manager.py`     | `stat()`, `rename()`, `mkstemp()`          |
| **Full Pipeline**     | `pipeline.py`         | **All combined + benchmarks**              |

## Quick Start

```bash
# Install dependencies
pip install pythainlp --break-system-packages

# Run optimized pipeline
python src/pipeline.py

# Run ALL benchmarks (for presentation)
python src/pipeline.py --all-benchmarks

# Run individual benchmarks
python src/data_loader.py          # mmap vs read
python src/preprocessor.py --benchmark  # 1 vs N workers
python src/sync_queue.py           # buffer sizes
python src/io_benchmark.py         # fsync strategies
python src/scheduler_demo.py       # CPU affinity
python src/file_manager.py         # page cache + atomic write
```

## Benchmark Results (on this dataset)

### Memory Management: mmap() vs read()
| Strategy        | Time   | Speedup |
|----------------|--------|---------|
| read() 64KB    | 0.067s | 1.00x   |
| mmap()         | 0.077s | 0.87x   |
| read() buffered| 0.082s | 0.82x   |
| read() 512B    | 0.267s | 0.25x   |

**Insight:** Chunk size matters more than mmap vs read for this file size (7MB).
mmap shines on larger files with random access patterns.

### I/O Management: fsync() frequency
| Strategy          | Time   | Rows/sec |
|-------------------|--------|----------|
| Buffered (8KB)    | 0.014s | 370,000  |
| fsync per 1000    | 0.057s | 88,000   |
| fsync per 100     | 0.136s | 36,700   |
| fsync per row     | 0.925s | 541      |

**Insight:** fsync() per row is ~68x slower than buffered. Batched fsync
(every 100-1000 rows) provides a good balance of durability and performance.

## Performance Trade-offs Discussion

1. **mmap vs read()**: mmap avoids kernel→user buffer copies (zero-copy) but
   incurs page fault overhead. For sequential reads of files <10MB, buffered
   read() with large chunks can match or beat mmap. mmap is superior for
   random access and files >100MB.

2. **Multiprocessing workers**: Speedup ≈ min(N, CPU_count). Each fork()
   creates a ~50MB overhead. Beyond CPU count, context switching dominates.

3. **fsync frequency**: Each fsync() waits for the disk controller to confirm
   physical write. Batching (every 100-1000 operations) gives 10-100x speedup
   over per-operation fsync while maintaining reasonable durability guarantees.

4. **Bounded buffer size**: The producer-consumer pattern with semaphores
   allows streaming processing. Small buffers cause frequent blocking (semaphore
   waits); large buffers reduce blocking but use more memory.

5. **CPU affinity**: Pinning workers to cores keeps L1/L2 cache warm, avoiding
   cold cache misses when the OS migrates processes between cores.

   ex: word from facebook เกาะพ่อแม่กิน วันๆอยู่แต่ในห้องชักเว่าจนตัวซีด เยี่ยวใส่ขวด นอนตีสามตื่นบ่ายสอง ตื่นมาชักเว่าต่อแล้วเล่นเกมกาชาทำเควสต์ฆ่าเวลาหลอกตัวเองไปเรื่อย จนดึก วนลูปไป
   Ngo ก็ Ngo อยู่วันยังค่ำ ลองศึกษาหาข้อมูลบ้างครับ มีคอมไว้แอคheeหรอ

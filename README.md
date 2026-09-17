# say.java
# Laboratory Benchmark: Monte Carlo Pi Estimation

CSS314_01N_03P_17.09.26_15:10
Akbota Kurman 230103276 
**Repository:** Shared with `@sufyanism`
## Part 1: The Phantom Bug (Data Race)

### Empirical Results (5 Runs)
* **Run 1:** Pi = 1.13356 (Hits: 14,169,447 / 50,000,000)
* **Run 2:** Pi = 1.06327 (Hits: 13,290,826 / 50,000,000)
* **Run 3:** Pi = 0.93634 (Hits: 11,704,267 / 50,000,000)
* **Run 4:** Pi = 1.08094 (Hits: 13,511,754 / 50,000,000)
* **Run 5:** Pi = 1.24263 (Hits: 15,532,902 / 50,000,000)

### Architectural Analysis
The operation `totalHits++` is not atomic. It expands into three separate operations: LOAD, ADD, and STORE. When 4 native Java threads execute this concurrently without synchronization, race conditions occur. Multiple threads read identical values from main memory, increment them locally, and overwrite each other's memory writes. This causes millions of hits to be lost, resulting in Pi collapsing far below 3.14159.

---

## Part 2: The Synchronization Trap

### Empirical Benchmarks
* **Single-Threaded Baseline:** 275 ms (Pi = 3.14158)
* **4 Threads (AtomicLong):** 922 ms (Pi = 3.14141)
* **Slowdown Factor:** 3.35x SLOWER than single-threaded baseline.

### Architectural Analysis
While `AtomicLong` restores mathematical accuracy (~3.14141), it forces threads to lock the hardware memory bus via `LOCK CMPXCHG` instructions on every iteration. Under the MESI cache coherency protocol, 4 cores continuously writing to the same L1 cache line cause non-stop Bus Invalidation signals. CPU cores spend over 90% of their execution cycles stalled on memory locks and cache misses.

---

## Part 3: OpenMP-Style Reduction

### Empirical Grid

| Threads (T) | Runtime (ms) | Speedup vs 1T | Efficiency (%) |
| :---: | :---: | :---: | :---: |
| **1** | 139 | 1.00x | 100.0% |
| **2** | 81 | 1.72x | 85.8% |
| **4** | 77 | 1.81x | 45.1% |
| **8** | 103 | 1.35x | 16.9% |
| **16** | 93 | 1.49x | 9.3% |
---

## Questions & Answers

### Q1: Look at your row for 16 threads. Why didn't your 8-core CPU run twice as fast as 8 threads?
**Answer:** An 8-core CPU contains exactly 8 physical execution pipelines. Running 16 threads relies on Simultaneous Multithreading (SMT / Hyper-Threading) to map 2 logical threads per physical core. Because SMT logical threads share execution units (ALUs, FPUs) and L1/L2 caches within the same physical core, and the Monte Carlo compute loop already saturates the hardware pipelines at 8 threads, allocating 16 threads cannot yield a 2x speedup. Runtime stays flat (~93 ms) due to thread context-switching overhead and resource contention.

### Q2: Why was the synchronized version in Part 2 slower than running on one single core?
**Answer:** Synchronized/atomic operations force sequential execution on shared memory using bus locks. Only one thread can perform the memory write at a time, while the others stall. Competing for the lock causes continuous kernel context switching, thread parking, and L1 cache invalidations (MESI protocol). A single thread executes continuously in CPU registers and L1 cache with zero lock contention, making it 3.35x faster than the atomic multi-threaded version.

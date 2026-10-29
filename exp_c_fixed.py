import time
from multiprocessing import Pool

def private_worker(iterations):
    # Each worker updates a private local variable; no shared lock is needed.
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum

def run_privatized(p, total_increments=20_000_000):
    base, remainder = divmod(total_increments, p)
    tasks = [base + (1 if i < remainder else 0) for i in range(p)]

    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # Combine results once, after workers finish
    elapsed = time.perf_counter() - t0

    assert total == total_increments, (
        f"Total mismatch: expected {total_increments}, got {total}"
    )
    return elapsed

def main():
    cores = [1, 2, 4]
    runtimes = {p: run_privatized(p) for p in cores}
    t1 = runtimes[1]
    print("Experiment C — Privatized State")
    for p in cores:
        tp = runtimes[p]
        if p == 1:
            print(f"P={p} | Runtime={tp:.3f}s | Speedup=1.00x | Karp-Flatt e=N/A")
        else:
            speedup = t1 / tp
            e = ((p * tp / t1) - 1.0) / (p - 1.0)
            efficiency = speedup / p * 100
            print(
                f"P={p} | Runtime={tp:.3f}s | Speedup={speedup:.2f}x | "
                f"Efficiency={efficiency:.1f}% | Karp-Flatt e={e:.3f}"
            )

if __name__ == "__main__":
    main()

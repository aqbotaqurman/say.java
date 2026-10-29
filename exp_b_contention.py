import time
from multiprocessing import Process, Value, Lock

def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1

def run_contended(p, total_increments=2_000_000):
    # Distribute all increments exactly, including any remainder.
    base, remainder = divmod(total_increments, p)
    counts = [base + (1 if i < remainder else 0) for i in range(p)]
    counter = Value('q', 0)
    lock = Lock()
    procs = [
        Process(target=contended_worker, args=(counter, lock, count))
        for count in counts
    ]

    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - t0

    assert counter.value == total_increments, (
        f"Counter mismatch: expected {total_increments}, got {counter.value}"
    )
    return elapsed

def main():
    cores = [1, 2, 4]
    runtimes = {p: run_contended(p) for p in cores}
    t1 = runtimes[1]
    print("Experiment B — Lock Contention")
    for p in cores:
        tp = runtimes[p]
        if p == 1:
            print(f"P={p} | Runtime={tp:.3f}s | Speedup=1.00x | Karp-Flatt e=N/A")
        else:
            speedup = t1 / tp
            e = ((p * tp / t1) - 1.0) / (p - 1.0)
            print(f"P={p} | Runtime={tp:.3f}s | Speedup={speedup:.2f}x | Karp-Flatt e={e:.3f}")

if __name__ == "__main__":
    main()

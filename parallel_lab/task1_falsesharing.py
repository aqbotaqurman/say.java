
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

STUDENT_ID = "230103276"
M = 16_000_000
P_PHYS = 8
RUNS = 7
MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (acc * MULTIPLIER + (i | 1)) & MASK
        arr[slot_idx] = acc

    del arr
    shm.close()


def fnv1a_hash(s):
    h = 14695981039346656037

    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & MASK

    return h


def run_trial(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    start = time.perf_counter_ns()

    try:
        processes = []

        for worker_id in range(p):
            iters = M // p
            slot_idx = worker_id * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        end = time.perf_counter_ns()

        return (end - start) / 1_000_000

    finally:
        shm.close()
        shm.unlink()


def measure(p, stride):
    samples = []

    for _ in range(RUNS):
        samples.append(run_trial(p, stride))

    samples.sort()

    median = statistics.median(samples)
    iqr = samples[5] - samples[1]

    throughput = M / (median * 1000)

    return median, iqr, throughput


def karp_flatt(speedup, p):
    if p == 1:
        return None

    return (1 / speedup - 1 / p) / (1 - 1 / p)


def main():
    print("===== PHASE 1: FALSE SHARING =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {mp.cpu_count()}")
    print(f"Total operations: {M:,}")
    print(f"Physical cores: {P_PHYS}")

    # PART 1: STRIDE SWEEP

    print("\n--- STRIDE SWEEP (p = 8) ---")
    print("Stride | Median ms | IQR ms | Mops/s")

    stride_results = {}

    for stride in [1, 2, 4, 8, 16, 32]:
        median, iqr, throughput = measure(P_PHYS, stride)

        stride_results[stride] = median

        print(
            f"{stride:6} | {median:9.3f} | "
            f"{iqr:7.3f} | {throughput:8.3f}"
        )

    # PART 2: PROCESS SCALING

    scaling_results = {}

    for stride in [1, 16]:
        print(f"\n--- PROCESS SCALING (stride = {stride}) ---")

        baseline = None
        scaling_results[stride] = {}

        print("p | Median ms | IQR ms | Speedup | Efficiency | Karp-Flatt")

        for p in [1, 2, 4, 8]:
            median, iqr, throughput = measure(p, stride)

            if p == 1:
                baseline = median

            speedup = baseline / median
            efficiency = speedup / p
            e = karp_flatt(speedup, p)

            scaling_results[stride][p] = median

            e_text = "-" if e is None else f"{e:.4f}"

            print(
                f"{p} | {median:9.3f} | {iqr:7.3f} | "
                f"{speedup:7.3f} | {efficiency:10.3f} | {e_text}"
            )

    # PART 3: VERIFICATION HASH

    m1 = stride_results[1]
    m2 = stride_results[16]
    hw = mp.cpu_count()

    signature_input = (
        f"PHASE1|{STUDENT_ID}|{hw}|{m1:.4f}|{m2:.4f}"
    )

    signature = fnv1a_hash(signature_input)

    print("\n--- PHASE 1 VERIFICATION ---")
    print(
        f"VERIFY_HASH [PHASE1]: {signature:016X} "
        f"(P_log ={hw})"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

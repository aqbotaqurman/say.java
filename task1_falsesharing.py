
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_SEED = 230103276
M = 24_000_000

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)

    processes = []

    try:
        iters = M // p

        for t in range(p):
            slot_idx = t * stride

            proc = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()

        for proc in processes:
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed with exit code {proc.exitcode}"
                )

    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()

        shm.close()
        shm.unlink()


def measure(p, stride):
    stats = run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup_runs=2,
        timed_runs=7
    )

    throughput = M / (stats.median_ms * 1000)

    return stats.median_ms, stats.iqr_ms, throughput


def main():
    print("PHASE 1 - FALSE SHARING ANALYSIS")
    print("=" * 65)
    print(f"Student ID: {STUDENT_SEED}")
    print(f"Total operations: {M:,}")
    print(f"Physical cores: 8")
    print()

    # TABLE 1A: Stride sensitivity at p = Pphys = 8
    print("TABLE 1A - STRIDE SENSITIVITY")
    print("-" * 65)
    print(
        f"{'Stride':>8} {'Bytes':>10} "
        f"{'Median ms':>15} {'IQR ms':>12} {'Mops/s':>12}"
    )

    stride_results = {}

    for stride in STRIDES:
        median, iqr, throughput = measure(8, stride)

        stride_results[stride] = median

        print(
            f"{stride:>8} {stride * 8:>10} "
            f"{median:>15.4f} {iqr:>12.4f} "
            f"{throughput:>12.4f}"
        )

    print()

    # TABLE 1B: Process scaling for stride 1 and stride 16
    all_results = {}

    for stride in [1, 16]:
        print(f"TABLE 1B - PROCESS SCALING (STRIDE = {stride})")
        print("-" * 75)
        print(
            f"{'p':>5} {'Median ms':>15} {'Speedup':>12} "
            f"{'Efficiency':>13} {'Karp-Flatt e(p)':>18}"
        )

        times = {}

        for p in PROCESS_COUNTS:
            median, iqr, throughput = measure(p, stride)
            times[p] = median

        t1 = times[1]

        for p in PROCESS_COUNTS:
            tp = times[p]
            speedup = t1 / tp
            efficiency = speedup / p

            if p >= 2:
                karp_flatt = (1 / speedup - 1 / p) / (1 - 1 / p)
                ef_text = f"{karp_flatt:.6f}"
            else:
                ef_text = "N/A"

            print(
                f"{p:>5} {tp:>15.4f} {speedup:>12.4f} "
                f"{efficiency:>13.4f} {ef_text:>18}"
            )

        all_results[stride] = times
        print()

    # Verification hash uses p = Pphys = 8
    median_stride_1 = stride_results[1]
    median_stride_16 = stride_results[16]

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_SEED,
            median_stride_1,
            median_stride_16
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()


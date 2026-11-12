
import multiprocessing as mp
from multiprocessing import shared_memory
import time

from harness import run_benchmark


P_PHYS = 8
BUFFER_SIZE = 96 * 1024 * 1024
FLOPS_PER_WORKER = 80_000_000

SHM_A = None
SHM_B = None
VIEW_A = None
VIEW_B = None


def initialize_memory(name_a, name_b):
    global SHM_A, SHM_B, VIEW_A, VIEW_B

    SHM_A = shared_memory.SharedMemory(name=name_a)
    SHM_B = shared_memory.SharedMemory(name=name_b)

    VIEW_A = memoryview(SHM_A.buf)
    VIEW_B = memoryview(SHM_B.buf)


def copy_worker(args):
    start, end = args

    VIEW_A[start:end] = VIEW_B[start:end]

    return end - start


def compute_worker(worker_id):
    a = 1.0000001
    b = 0.0000001

    x0 = 0.1
    x1 = 0.2
    x2 = 0.3
    x3 = 0.4
    x4 = 0.5
    x5 = 0.6
    x6 = 0.7
    x7 = 0.8
    x8 = 0.9
    x9 = 1.0
    x10 = 1.1
    x11 = 1.2
    x12 = 1.3
    x13 = 1.4
    x14 = 1.5
    x15 = 1.6

    iterations = FLOPS_PER_WORKER // 32

    for _ in range(iterations):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return (
        x0 + x1 + x2 + x3 +
        x4 + x5 + x6 + x7 +
        x8 + x9 + x10 + x11 +
        x12 + x13 + x14 + x15
    )


def main():
    print("PHASE 3A - ROOFLINE CHARACTERIZATION")
    print("=" * 65)
    print(f"Physical workers: {P_PHYS}")
    print(f"Buffer size: {BUFFER_SIZE / (1024 * 1024):.0f} MB")
    print()

    shm_a = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=BUFFER_SIZE
    )

    try:
        # Initialize source buffer once, outside the timed region
        source_view = memoryview(shm_b.buf)
        source_view[:] = b"\x01" * BUFFER_SIZE
        source_view.release()

        chunk_size = BUFFER_SIZE // P_PHYS
        chunks = []

        for worker_id in range(P_PHYS):
            start = worker_id * chunk_size
            end = (
                (worker_id + 1) * chunk_size
                if worker_id < P_PHYS - 1
                else BUFFER_SIZE
            )
            chunks.append((start, end))

        with mp.Pool(
            processes=P_PHYS,
            initializer=initialize_memory,
            initargs=(shm_a.name, shm_b.name)
        ) as pool:

            def memory_test():
                pool.map(copy_worker, chunks)

            memory_stats = run_benchmark(
                memory_test,
                warmup_runs=2,
                timed_runs=7
            )

        bandwidth = 192.0 / memory_stats.median_ms

        print("MEMORY BANDWIDTH")
        print(f"Median time: {memory_stats.median_ms:.4f} ms")
        print(f"IQR: {memory_stats.iqr_ms:.4f} ms")
        print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")
        print()

        # Compute ceiling benchmark
        with mp.Pool(processes=P_PHYS) as pool:

            def compute_test():
                pool.map(compute_worker, range(P_PHYS))

            compute_stats = run_benchmark(
                compute_test,
                warmup_runs=2,
                timed_runs=7
            )

        total_flops = P_PHYS * FLOPS_PER_WORKER

        compute_ceiling = (
            total_flops /
            (compute_stats.median_ms * 1_000_000)
        )

        ridge_point = compute_ceiling / bandwidth

        print("PYTHON COMPUTE CEILING")
        print(f"Median time: {compute_stats.median_ms:.4f} ms")
        print(f"IQR: {compute_stats.iqr_ms:.4f} ms")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"Compute ceiling: {compute_ceiling:.4f} GFLOP/s")
        print()

        print("EMPIRICAL ROOFLINE")
        print(f"Bandwidth: {bandwidth:.4f} GB/s")
        print(f"Compute ceiling: {compute_ceiling:.4f} GFLOP/s")
        print(f"Ridge point: {ridge_point:.4f} FLOPs/Byte")

    finally:
        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()


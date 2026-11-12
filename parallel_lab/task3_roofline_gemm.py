
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import platform
import statistics
import time
import hashlib

# ============================================================
# PHASE 3: EMPIRICAL ROOFLINE AND CACHE-BLOCKED GEMM
# Student ID: 230103276
# Apple M2: 8 logical processors
# ============================================================

STUDENT_ID = "230103276"
P = 8

# Phase 3A: memory bandwidth
BUFFER_SIZE = 64 * 1024 * 1024  # 64 MiB per buffer
COPY_REPEATS = 5

# Phase 3A: compute ceiling
FLOPS_PER_WORKER = 60_000_000
FMA_VARIABLES = 16
FMA_FLOPS_PER_LOOP = FMA_VARIABLES * 2
FMA_ITERATIONS = FLOPS_PER_WORKER // FMA_FLOPS_PER_LOOP
COMPUTE_REPEATS = 3

# Phase 3B: matrix multiplication
N = 256
GEMM_REPEATS = 3
BLOCK_SIZES = [8, 16, 32, 64, 96, 128]


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------

def median_iqr(values):
    ordered = sorted(values)
    med = statistics.median(ordered)
    q1 = statistics.median(ordered[:len(ordered) // 2])
    q3 = statistics.median(ordered[(len(ordered) + 1) // 2:])
    return med, q3 - q1


def fnv1a_64(text):
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016X}"


# ------------------------------------------------------------
# PHASE 3A — DRAM bandwidth
# ------------------------------------------------------------

def copy_worker(task):
    name_a, name_b, start, end = task

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    view_a = shm_a.buf
    view_b = shm_b.buf

    view_a[start:end] = view_b[start:end]

    view_a.release()
    view_b.release()
    shm_a.close()
    shm_b.close()

    return end - start


def measure_bandwidth():
    print("\n===== PHASE 3A: DRAM BANDWIDTH =====")
    print("Allocating two 64 MiB shared-memory buffers...")

    shm_a = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE)
    shm_b = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE)

    # Initialize the source buffer so the measurement reads real data.
    block = bytes([1]) * (1024 * 1024)
    for offset in range(0, BUFFER_SIZE, len(block)):
        shm_b.buf[offset:offset + len(block)] = block

    chunk = BUFFER_SIZE // P
    tasks = [
        (shm_a.name, shm_b.name, i * chunk, (i + 1) * chunk)
        for i in range(P)
    ]

    times = []

    try:
        with mp.Pool(processes=P) as pool:
            pool.map(copy_worker, tasks)  # warm-up

            for trial in range(COPY_REPEATS):
                start = time.perf_counter()
                pool.map(copy_worker, tasks)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"Copy trial {trial + 1}: {elapsed * 1000:.3f} ms")

        median_s, iqr_s = median_iqr(times)

        # Total traffic: 64 MB read + 64 MB written.
        bandwidth_gbs = (2 * 64 * 10**6) / (median_s * 10**9)

        print(f"\nMedian copy time: {median_s * 1000:.3f} ms")
        print(f"IQR: {iqr_s * 1000:.3f} ms")
        print(f"Sustained DRAM bandwidth: {bandwidth_gbs:.3f} GB/s")

        return bandwidth_gbs

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()


# ------------------------------------------------------------
# PHASE 3A — Compute ceiling using 16 independent FMA variables
# ------------------------------------------------------------

def fma_worker(_):
    fma = math.fma

    x0 = 0.11
    x1 = 0.12
    x2 = 0.13
    x3 = 0.14
    x4 = 0.15
    x5 = 0.16
    x6 = 0.17
    x7 = 0.18
    x8 = 0.19
    x9 = 0.20
    x10 = 0.21
    x11 = 0.22
    x12 = 0.23
    x13 = 0.24
    x14 = 0.25
    x15 = 0.26

    a = 1.00000001
    b = 0.0000001

    for _ in range(FMA_ITERATIONS):
        x0 = fma(x0, a, b)
        x1 = fma(x1, a, b)
        x2 = fma(x2, a, b)
        x3 = fma(x3, a, b)
        x4 = fma(x4, a, b)
        x5 = fma(x5, a, b)
        x6 = fma(x6, a, b)
        x7 = fma(x7, a, b)
        x8 = fma(x8, a, b)
        x9 = fma(x9, a, b)
        x10 = fma(x10, a, b)
        x11 = fma(x11, a, b)
        x12 = fma(x12, a, b)
        x13 = fma(x13, a, b)
        x14 = fma(x14, a, b)
        x15 = fma(x15, a, b)

    return (
        x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
        + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
    )


def measure_compute_ceiling():
    print("\n===== PHASE 3A: COMPUTE CEILING =====")
    print(f"FMA iterations per worker: {FMA_ITERATIONS:,}")
    print(f"FLOPs per worker: {FLOPS_PER_WORKER:,}")
    print(f"Workers: {P}")

    times = []

    with mp.Pool(processes=P) as pool:
        pool.map(fma_worker, range(P))  # warm-up

        for trial in range(COMPUTE_REPEATS):
            start = time.perf_counter()
            results = pool.map(fma_worker, range(P))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"Compute trial {trial + 1}: {elapsed:.3f} s")

    median_s, iqr_s = median_iqr(times)

    total_flops = FLOPS_PER_WORKER * P
    compute_gflops = total_flops / (median_s * 1e9)

    print(f"\nMedian compute time: {median_s:.3f} s")
    print(f"IQR: {iqr_s:.3f} s")
    print(f"Sustained compute ceiling: {compute_gflops:.4f} GFLOP/s")

    return compute_gflops


# ------------------------------------------------------------
# PHASE 3B — Shared-memory GEMM worker
# ------------------------------------------------------------

def gemm_worker(task):
    name_a, name_b, name_c, n, start_row, end_row, variant, bs = task

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    A = (ctypes.c_double * (n * n)).from_buffer(shm_a.buf)
    B = (ctypes.c_double * (n * n)).from_buffer(shm_b.buf)
    C = (ctypes.c_double * (n * n)).from_buffer(shm_c.buf)

    if variant == 1:
        # Variant 1: naive i-j-k.
        # B is accessed with a stride of N in the inner loop.
        for i in range(start_row, end_row):
            for j in range(n):
                total = 0.0
                for k in range(n):
                    total += A[i * n + k] * B[k * n + j]
                C[i * n + j] = total

    elif variant == 2:
        # Variant 2: reordered i-k-j.
        # j is the innermost loop, giving unit-stride access.
        for i in range(start_row, end_row):
            for k in range(n):
                a_ik = A[i * n + k]
                b_base = k * n
                c_base = i * n
                for j in range(n):
                    C[c_base + j] += a_ik * B[b_base + j]

    elif variant == 3:
        # Variant 3: cache-blocked six-loop GEMM.
        # Outer loops traverse tiles; inner micro-kernel uses i-k-j.
        for ii in range(start_row, end_row, bs):
            i_end = min(ii + bs, end_row)

            for jj in range(0, n, bs):
                j_end = min(jj + bs, n)

                for kk in range(0, n, bs):
                    k_end = min(kk + bs, n)

                    for i in range(ii, i_end):
                        c_base = i * n
                        for k in range(kk, k_end):
                            a_ik = A[i * n + k]
                            b_base = k * n

                            for j in range(jj, j_end):
                                C[c_base + j] += a_ik * B[b_base + j]

    else:
        raise ValueError("Unknown GEMM variant")

    # Release ctypes views before closing shared memory.
    del A, B, C
    shm_a.close()
    shm_b.close()
    shm_c.close()

    return end_row - start_row


# ------------------------------------------------------------
# PHASE 3B — GEMM helpers
# ------------------------------------------------------------

def clear_matrix(C, size):
    for i in range(size):
        C[i] = 0.0


def run_gemm_variant(pool, names, variant, bs=None):
    name_a, name_b, name_c = names
    rows_per_worker = N // P

    tasks = [
        (
            name_a,
            name_b,
            name_c,
            N,
            i * rows_per_worker,
            (i + 1) * rows_per_worker,
            variant,
            bs
        )
        for i in range(P)
    ]

    times = []

    for _ in range(GEMM_REPEATS):
        # Reset output matrix before each measurement.
        shm_c = shared_memory.SharedMemory(name=name_c)
        C = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)
        clear_matrix(C, N * N)
        del C
        shm_c.close()

        start = time.perf_counter()
        pool.map(gemm_worker, tasks)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_s, iqr_s = median_iqr(times)
    operations = 2 * (N ** 3)
    gflops = operations / (median_s * 1e9)

    return median_s, iqr_s, gflops


def run_gemm():
    print("\n===== PHASE 3B: GEMM =====")
    print(f"Matrix size: {N} x {N}")
    print(f"Processes: {P}")
    print(f"Repeats per variant: {GEMM_REPEATS}")

    size = N * N

    shm_a = shared_memory.SharedMemory(
        create=True, size=size * ctypes.sizeof(ctypes.c_double)
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=size * ctypes.sizeof(ctypes.c_double)
    )
    shm_c = shared_memory.SharedMemory(
        create=True, size=size * ctypes.sizeof(ctypes.c_double)
    )

    A = (ctypes.c_double * size).from_buffer(shm_a.buf)
    B = (ctypes.c_double * size).from_buffer(shm_b.buf)
    C = (ctypes.c_double * size).from_buffer(shm_c.buf)

    # Deterministic small values for numerical stability.
    for i in range(size):
        A[i] = ((i % 17) - 8) * 0.01
        B[i] = ((i % 13) - 6) * 0.01
        C[i] = 0.0

    names = (shm_a.name, shm_b.name, shm_c.name)

    # Release parent ctypes views before workers access the buffers.
    del A, B, C

    results = {}
    best_block = None
    best_block_gflops = -1.0
    reference_result = None

    try:
        with mp.Pool(processes=P) as pool:
            # Variant 1
            t, iqr, gflops = run_gemm_variant(pool, names, 1)
            results["variant1"] = (t, iqr, gflops)
            print(
                f"Variant 1 (i-j-k): median={t:.4f} s, "
                f"IQR={iqr:.4f} s, GFLOP/s={gflops:.4f}"
            )

            # Variant 2
            t, iqr, gflops = run_gemm_variant(pool, names, 2)
            results["variant2"] = (t, iqr, gflops)
            print(
                f"Variant 2 (i-k-j): median={t:.4f} s, "
                f"IQR={iqr:.4f} s, GFLOP/s={gflops:.4f}"
            )

            # Save Variant 2 output as the correctness reference.
            shm_c_check = shared_memory.SharedMemory(name=names[2])
            C = (ctypes.c_double * size).from_buffer(shm_c_check.buf)
            reference_result = list(C)
            del C
            shm_c_check.close()           
            C = (ctypes.c_double * size).from_buffer(shm_c.buf)
            reference_result = list(C)
            del C
            shm_c.close()

            # Variant 3: block-size sweep.
            print("\nCache-block size sweep:")
            for bs in BLOCK_SIZES:
                t, iqr, gflops = run_gemm_variant(
                    pool, names, 3, bs
                )

                # Compare current blocked result with Variant 2.
                check_shm = shared_memory.SharedMemory(name=names[2])
                check_c = (ctypes.c_double * size).from_buffer(check_shm.buf)

                max_error = max(
                    abs(check_c[i] - reference_result[i])
                    for i in range(size)
                )

                del check_c
                check_shm.close()

                tile_footprint = 3 * (bs ** 2) * 8
                intensity = bs / 12.0

                results[f"BS{bs}"] = (t, iqr, gflops, max_error)

                print(
                    f"BS={bs:3d} | median={t:.4f} s | "
                    f"IQR={iqr:.4f} s | GFLOP/s={gflops:.4f} | "
                    f"tile={tile_footprint:,} B | "
                    f"I={intensity:.3f} FLOP/B | "
                    f"max error={max_error:.3e}"
                )

                if gflops > best_block_gflops:
                    best_block_gflops = gflops
                    best_block = bs

                if max_error >= 1e-9:
                    print(
                        "WARNING: correctness tolerance was not met "
                        "for this block size."
                    )

        variant1_gflops = results["variant1"][2]

        print("\n===== GEMM SUMMARY =====")
        print(f"Best block size BS*: {best_block}")
        print(f"Best blocked GFLOP/s: {best_block_gflops:.4f}")
        print(f"Variant 1 GFLOP/s: {variant1_gflops:.4f}")

        return (
            variant1_gflops,
            best_block_gflops,
            best_block,
            results
        )

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_c.close()
        shm_c.unlink()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("===================================================")
    print("PHASE 3 — EMPIRICAL ROOFLINE & CACHE-BLOCKED GEMM")
    print("===================================================")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical processes: {P}")

    if not hasattr(math, "fma"):
        raise RuntimeError(
            "This Python version does not provide math.fma."
        )

    # Part 3A
    bandwidth = measure_bandwidth()
    compute_gflops = measure_compute_ceiling()

    ridge_point = compute_gflops / bandwidth

    print("\n===== ROOFLINE PARAMETERS =====")
    print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")
    print(f"Sustained compute: {compute_gflops:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    # Part 3B
    variant1_gflops, best_gflops, best_bs, gemm_results = run_gemm()

    # Final verification hash for Phase 3.
    hash_input = (
        f"PHASE3|{STUDENT_ID}|{P}|"
        f"{variant1_gflops:.4f}|{best_gflops:.4f}"
    )
    verify_hash = fnv1a_64(hash_input)

    print("\n===== PHASE 3 FINAL RESULTS =====")
    print(f"GFLOP/s Variant 1: {variant1_gflops:.4f}")
    print(f"GFLOP/s Variant 3 (BS*={best_bs}): {best_gflops:.4f}")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")
    print(f"VERIFY_HASH [PHASE3]: {verify_hash} (P_log = {P})")


if __name__ == "__main__":
    mp.freeze_support()
    main()


import multiprocessing as mp
import math
import platform
import time

# ============================================================
# PHASE 4: SKEWED WORKLOADS & DYNAMIC CHUNK QUEUE
# Student ID: 230103276
# ============================================================

STUDENT_ID = "230103276"
P = 8
N = 8000
CHUNK_SIZES = [1, 8, 64, 512]

# Shared objects initialized once in each worker process.
_ACTIVE_TIMES = None
_QUEUE_COUNTER = None


# ------------------------------------------------------------
# FNV-1a verification hash
# ------------------------------------------------------------

def fnv1a_64(text):
    value = 0xCBF29CE484222325

    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    return f"{value:016X}"


# ------------------------------------------------------------
# Workload definition
# ------------------------------------------------------------

def item_iterations(i):
    return 1 + (i * i) // 1600


def process_item(i):
    """
    Execute K(i) heavy iterations.
    Each iteration performs sin, cos, and sqrt.
    """
    iterations = item_iterations(i)
    x = (i + 1) * 0.001

    for _ in range(iterations):
        x = (
            math.sin(x)
            + math.cos(x)
            + math.sqrt(abs(x) + 1.0)
        )

    return x


# ------------------------------------------------------------
# Worker initialization
# ------------------------------------------------------------

def init_worker(active_times, queue_counter):
    global _ACTIVE_TIMES, _QUEUE_COUNTER

    _ACTIVE_TIMES = active_times
    _QUEUE_COUNTER = queue_counter


# ------------------------------------------------------------
# Worker execution
# ------------------------------------------------------------

def scheduling_worker(args):
    worker_id, strategy, chunk_size, n, p = args

    start_time = time.perf_counter()
    checksum = 0.0

    if strategy == "block":
        # Contiguous Block Static:
        # Each worker receives one continuous range.
        start = worker_id * n // p
        end = (worker_id + 1) * n // p

        for i in range(start, end):
            checksum += process_item(i)

    elif strategy == "cyclic":
        # Cyclic Round-Robin Static:
        # Workers process every p-th item.
        for i in range(worker_id, n, p):
            checksum += process_item(i)

    elif strategy == "dynamic":
        # Shared-Memory Dynamic Queue:
        # Workers obtain the next chunk using a synchronized counter.
        while True:
            with _QUEUE_COUNTER.get_lock():
                start = _QUEUE_COUNTER.value
                _QUEUE_COUNTER.value += chunk_size

            if start >= n:
                break

            end = min(start + chunk_size, n)

            for i in range(start, end):
                checksum += process_item(i)

    else:
        raise ValueError("Unknown scheduling strategy")

    elapsed = time.perf_counter() - start_time

    # Each worker writes only to its own shared-array position.
    _ACTIVE_TIMES[worker_id] = elapsed

    return checksum


# ------------------------------------------------------------
# Measurement and imbalance
# ------------------------------------------------------------

def reset_active_times(active_times):
    for i in range(P):
        active_times[i] = 0.0


def calculate_imbalance(active_times):
    values = list(active_times)

    average = sum(values) / len(values)

    if average == 0:
        return 0.0

    return max(values) / average


def run_strategy(pool, active_times, strategy, chunk_size=None):
    reset_active_times(active_times)

    tasks = [
        (worker_id, strategy, chunk_size, N, P)
        for worker_id in range(P)
    ]

    start = time.perf_counter()
    checksums = pool.map(scheduling_worker, tasks)
    wall_time = time.perf_counter() - start

    worker_times = list(active_times)
    imbalance = calculate_imbalance(worker_times)

    # Consume results so all worker computations are accounted for.
    checksum_total = sum(checksums)

    return wall_time, worker_times, imbalance, checksum_total


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    print("===================================================")
    print("PHASE 4 — SKEWED WORKLOADS & DYNAMIC CHUNK QUEUE")
    print("===================================================")

    print(f"Student ID: {STUDENT_ID}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical processes: {P}")
    print(f"Number of items: {N}")

    print("\nWorkload:")
    print("K(i) = 1 + floor((i / 40)^2)")
    print("Operations: sin, cos, sqrt")

    # Shared active runtime for each worker.
    active_times = mp.Array("d", P, lock=False)

    # Synchronized shared index counter for dynamic scheduling.
    queue_counter = mp.Value("i", 0, lock=True)

    results = {}

    # Pool creation is outside the timed measurements.
    with mp.Pool(
        processes=P,
        initializer=init_worker,
        initargs=(active_times, queue_counter)
    ) as pool:

        # ----------------------------------------------------
        # Strategy 1: Contiguous Block Static
        # ----------------------------------------------------
        print("\n===== STRATEGY 1: CONTIGUOUS BLOCK STATIC =====")

        wall, worker_times, imbalance, checksum = run_strategy(
            pool, active_times, "block"
        )

        results["block"] = {
            "wall": wall,
            "workers": worker_times,
            "imbalance": imbalance
        }

        print(f"Wall time: {wall:.4f} s")
        print("Worker active times (s):")
        for i, value in enumerate(worker_times):
            print(f"  Worker {i}: {value:.4f}")

        print(f"Load Imbalance Factor (lambda): {imbalance:.4f}")

        # ----------------------------------------------------
        # Strategy 2: Cyclic Round-Robin Static
        # ----------------------------------------------------
        print("\n===== STRATEGY 2: CYCLIC ROUND-ROBIN STATIC =====")

        wall, worker_times, imbalance, checksum = run_strategy(
            pool, active_times, "cyclic"
        )

        results["cyclic"] = {
            "wall": wall,
            "workers": worker_times,
            "imbalance": imbalance
        }

        print(f"Wall time: {wall:.4f} s")
        print("Worker active times (s):")
        for i, value in enumerate(worker_times):
            print(f"  Worker {i}: {value:.4f}")

        print(f"Load Imbalance Factor (lambda): {imbalance:.4f}")

        # ----------------------------------------------------
        # Strategy 3: Dynamic Queue
        # ----------------------------------------------------
        print("\n===== STRATEGY 3: DYNAMIC CHUNK QUEUE =====")

        for chunk_size in CHUNK_SIZES:
            # Reset shared queue before each dynamic experiment.
            with queue_counter.get_lock():
                queue_counter.value = 0

            wall, worker_times, imbalance, checksum = run_strategy(
                pool,
                active_times,
                "dynamic",
                chunk_size
            )

            results[chunk_size] = {
                "wall": wall,
                "workers": worker_times,
                "imbalance": imbalance
            }

            print(f"\nChunk size C = {chunk_size}")
            print(f"Wall time: {wall:.4f} s")
            print("Worker active times (s):")

            for i, value in enumerate(worker_times):
                print(f"  Worker {i}: {value:.4f}")

            print(f"Load Imbalance Factor (lambda): {imbalance:.4f}")

    # --------------------------------------------------------
    # Find optimal dynamic chunk size
    # --------------------------------------------------------

    best_chunk = min(
        CHUNK_SIZES,
        key=lambda c: results[c]["wall"]
    )

    block_wall = results["block"]["wall"]
    dynamic_wall = results[best_chunk]["wall"]

    print("\n===================================================")
    print("PHASE 4 SUMMARY")
    print("===================================================")

    print("\nStrategy                 Wall Time (s)    Lambda")

    print(
        f"Block Static             "
        f"{block_wall:12.4f}    "
        f"{results['block']['imbalance']:.4f}"
    )

    print(
        f"Cyclic Static            "
        f"{results['cyclic']['wall']:12.4f}    "
        f"{results['cyclic']['imbalance']:.4f}"
    )

    for chunk_size in CHUNK_SIZES:
        print(
            f"Dynamic C={chunk_size:<3}          "
            f"{results[chunk_size]['wall']:12.4f}    "
            f"{results[chunk_size]['imbalance']:.4f}"
        )

    print(f"\nOptimal dynamic chunk C*: {best_chunk}")
    print(f"Block Static wall time: {block_wall:.4f} s")
    print(f"Best Dynamic wall time: {dynamic_wall:.4f} s")

    if dynamic_wall > 0:
        improvement = (
            (block_wall - dynamic_wall) / block_wall
        ) * 100
    else:
        improvement = 0.0

    print(f"Dynamic time improvement: {improvement:.2f}%")

    # --------------------------------------------------------
    # Phase 4 verification hash
    # --------------------------------------------------------

    hash_input = (
        f"PHASE4|{STUDENT_ID}|{P}|"
        f"{block_wall:.4f}|{dynamic_wall:.4f}"
    )

    verify_hash = fnv1a_64(hash_input)

    print("\n===== PHASE 4 FINAL RESULTS =====")
    print(f"Block Static Twall: {block_wall:.4f} s")
    print(f"Dynamic Twall (C*={best_chunk}): {dynamic_wall:.4f} s")
    print(f"VERIFY_HASH [PHASE4]: {verify_hash} (P_log = {P})")


if __name__ == "__main__":
    mp.freeze_support()
    main()

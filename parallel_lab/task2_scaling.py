
import multiprocessing as mp
import math
import time
import statistics

STUDENT_ID = "230103276"
P_LOG = 8

STRONG_N = 1_200_000
WEAK_N0 = 200_000

RUNS = 7
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_hash(s):
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & MASK
    return h


# Stage A: serial work in the parent process
def stage_a(n):
    result = 0.0

    for i in range(n):
        x = i * 0.00001 + 1.0

        for _ in range(10):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)

        result += x

    return result


# Stage B: parallel worker
def stage_b_worker(task):
    start, end = task
    result = 0.0

    for i in range(start, end):
        x = i * 0.00001 + 1.0

        for _ in range(30):
            x = math.fma(
                math.sin(x),
                0.5,
                math.sqrt(abs(x) + 1.0)
            )

        result += x

    return result


def make_chunks(n, p):
    chunk_size = (n + p - 1) // p

    return [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]


def run_pipeline(pool, n, p):
    stage_a_n = int(0.05 * n)

    total_start = time.perf_counter_ns()

    # Stage A: serial
    a_start = time.perf_counter_ns()
    result_a = stage_a(stage_a_n)
    a_end = time.perf_counter_ns()

    stage_a_ms = (a_end - a_start) / 1e6

    # Stage B: parallel
    chunks = make_chunks(n, p)
    results = pool.map(stage_b_worker, chunks)

    result_b = sum(results)
    total_end = time.perf_counter_ns()

    total_ms = (total_end - total_start) / 1e6

    # Keep the calculated results observable
    checksum = result_a + result_b

    return total_ms, stage_a_ms, checksum


def run_experiment(n, p):
    total_times = []
    stage_a_times = []

    # Pool is created outside the timed trials
    with mp.Pool(processes=p) as pool:

        for _ in range(RUNS):
            total_ms, stage_a_ms, checksum = run_pipeline(
                pool, n, p
            )

            total_times.append(total_ms)
            stage_a_times.append(stage_a_ms)

    total_median = statistics.median(total_times)
    stage_a_median = statistics.median(stage_a_times)

    sorted_times = sorted(total_times)
    iqr = sorted_times[5] - sorted_times[1]

    return total_median, iqr, stage_a_median


def main():

    print("===== PHASE 2: STRONG AND WEAK SCALING =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical processors: {mp.cpu_count()}")

    # =========================================
    # PART 1: STRONG SCALING
    # =========================================

    print("\n--- STRONG SCALING ---")
    print("p | Median ms | IQR ms | Speedup | Efficiency | Stage A ms")

    strong_results = {}
    baseline = None

    for p in range(1, P_LOG + 1):

        median, iqr, stage_a_ms = run_experiment(STRONG_N, p)

        if p == 1:
            baseline = median

        speedup = baseline / median
        efficiency = speedup / p

        strong_results[p] = (median, stage_a_ms)

        print(
            f"{p} | {median:.4f} | {iqr:.4f} | "
            f"{speedup:.4f} | {efficiency:.4f} | "
            f"{stage_a_ms:.4f}"
        )

    # Amdahl's law
    serial_fraction = strong_results[1][1] / baseline
    amdahl_cap = 1 / serial_fraction

    strong_speedup_log = baseline / strong_results[P_LOG][0]

    print(f"\nSerial fraction: {serial_fraction:.6f}")
    print(f"Amdahl's theoretical cap: {amdahl_cap:.4f}")
    print(f"Observed speedup at p={P_LOG}: {strong_speedup_log:.4f}")

    # =========================================
    # PART 2: WEAK SCALING
    # =========================================

    print("\n--- WEAK SCALING ---")
    print("p | N | Median ms | IQR ms | Weak Efficiency | Gustafson")

    weak_results = {}
    weak_baseline = None
    gustafson_log = None

    for p in range(1, P_LOG + 1):

        n = p * WEAK_N0

        median, iqr, stage_a_ms = run_experiment(n, p)

        if p == 1:
            weak_baseline = median

        efficiency = weak_baseline / median

        alpha = stage_a_ms / median

        gustafson = p - alpha * (p - 1)

        weak_results[p] = (median, stage_a_ms)

        print(
            f"{p} | {n} | {median:.4f} | {iqr:.4f} | "
            f"{efficiency:.4f} | {gustafson:.4f}"
        )

        if p == P_LOG:
            gustafson_log = gustafson

    # =========================================
    # PART 3: VERIFICATION HASH
    # =========================================

    signature_input = (
        f"PHASE2|{STUDENT_ID}|{P_LOG}|"
        f"{strong_speedup_log:.4f}|{gustafson_log:.4f}"
    )

    signature = fnv1a_hash(signature_input)

    print("\n--- PHASE 2 VERIFICATION ---")

    print(
        f"VERIFY_HASH [PHASE2]: {signature:016X} "
        f"(P_log ={P_LOG})"
    )

    print("\n--- ARCHITECTURAL INFLECTION ---")

    print(
        "Pphys = 8 and Plog = 8. "
        "The p=8 to p=9 transition cannot be measured "
        "on this configuration."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()

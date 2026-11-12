
import multiprocessing as mp
import math
import time
import statistics

from harness import run_benchmark, emit_verify_tag


STUDENT_SEED = 230103276

P_LOG = 8
P_VALUES = list(range(1, P_LOG + 1))

N_STRONG = 1_600_000
N0_WEAK = 250_000


def serial_stage(n):
    """Serial Stage A: 5% of N, with 12 math operations per element."""
    count = int(0.05 * n)
    result = 0.0

    for i in range(count):
        x = (i % 1000 + 1) * 0.001

        for _ in range(6):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)

        result += x

    return result


def parallel_worker(args):
    """Stage B: 35 multiply-add and trigonometric iterations per element."""
    start, end = args
    result = 0.0

    for i in range(start, end):
        x = (i % 1000 + 1) * 0.001

        for _ in range(35):
            x = x * 1.0000001 + math.sin(x)

        result += x

    return result


def make_chunks(n, p):
    chunks = []

    for worker_id in range(p):
        start = worker_id * n // p
        end = (worker_id + 1) * n // p
        chunks.append((start, end))

    return chunks


def run_pipeline(n, pool, p, stage_a_samples):
    # Stage A runs in the parent process
    start_a = time.perf_counter_ns()
    serial_result = serial_stage(n)
    end_a = time.perf_counter_ns()

    stage_a_ms = (end_a - start_a) / 1e6
    stage_a_samples.append(stage_a_ms)

    # Stage B runs across persistent worker processes
    chunks = make_chunks(n, p)
    results = pool.map(parallel_worker, chunks)

    # Consume results to ensure the work is completed
    checksum = serial_result + sum(results)

    return checksum


def benchmark_configuration(n, p):
    stage_a_samples = []
    chunks = make_chunks(n, p)

    with mp.Pool(processes=p) as pool:

        def experiment():
            run_pipeline(n, pool, p, stage_a_samples)

        stats = run_benchmark(
            experiment,
            warmup_runs=2,
            timed_runs=7
        )

    stage_a_median = statistics.median(stage_a_samples)

    return stats.median_ms, stats.iqr_ms, stage_a_median


def main():
    print("PHASE 2 - STRONG AND WEAK SCALING")
    print("=" * 85)
    print(f"Student ID: {STUDENT_SEED}")
    print(f"Logical processors: {P_LOG}")
    print(f"Strong scaling N: {N_STRONG:,}")
    print(f"Weak scaling N0: {N0_WEAK:,}")
    print()

    # --------------------------------------------------
    # EXPERIMENT 2A: STRONG SCALING
    # --------------------------------------------------

    print("TABLE 2A - STRONG SCALING")
    print("-" * 85)
    print(
        f"{'p':>4} {'Median ms':>15} {'IQR ms':>12} "
        f"{'Speedup':>12} {'Efficiency':>13} {'Stage A ms':>14}"
    )

    strong_times = {}
    strong_stage_a = {}
    strong_speedups = {}

    for p in P_VALUES:
        median, iqr, stage_a = benchmark_configuration(
            N_STRONG, p
        )

        strong_times[p] = median
        strong_stage_a[p] = stage_a

        print(
            f"{p:>4} {median:>15.4f} {iqr:>12.4f} "
            f"{'':>12} {'':>13} {stage_a:>14.4f}"
        )

    t1_strong = strong_times[1]

    print()
    print("Calculated strong scaling values:")

    for p in P_VALUES:
        speedup = t1_strong / strong_times[p]
        efficiency = speedup / p
        strong_speedups[p] = speedup

        print(
            f"p={p}: Speedup={speedup:.4f}, "
            f"Efficiency={efficiency:.4f}"
        )

    print()

    # --------------------------------------------------
    # AMDAHL'S LAW
    # --------------------------------------------------

    serial_fraction = strong_stage_a[1] / t1_strong

    if serial_fraction > 0:
        amdahl_max = 1.0 / serial_fraction
        amdahl_p8 = 1.0 / (
            serial_fraction + (1.0 - serial_fraction) / P_LOG
        )
    else:
        amdahl_max = float("inf")
        amdahl_p8 = float("inf")

    print("AMDAHL'S LAW")
    print("-" * 85)
    print(f"Serial fraction s: {serial_fraction:.6f}")
    print(f"Theoretical maximum speedup: {amdahl_max:.4f}")
    print(f"Theoretical speedup at p={P_LOG}: {amdahl_p8:.4f}")
    print(f"Observed speedup at p={P_LOG}: {strong_speedups[P_LOG]:.4f}")
    print()

    # Marginal speedup gain around the M2 P-core/E-core transition.
    # This is an observed process-count comparison, not CPU pinning.
    if 3 in strong_speedups and 4 in strong_speedups and 5 in strong_speedups:
        gain_3_to_4 = strong_speedups[4] - strong_speedups[3]
        gain_4_to_5 = strong_speedups[5] - strong_speedups[4]

        if gain_3_to_4 != 0:
            drop_percent = (
                (gain_3_to_4 - gain_4_to_5) / gain_3_to_4
            ) * 100
        else:
            drop_percent = 0.0

        print("MARGINAL SPEEDUP ANALYSIS")
        print(f"Gain from p=3 to p=4: {gain_3_to_4:.4f}")
        print(f"Gain from p=4 to p=5: {gain_4_to_5:.4f}")
        print(f"Change in marginal gain: {drop_percent:.2f}%")
        print()

    # --------------------------------------------------
    # EXPERIMENT 2B: WEAK SCALING
    # --------------------------------------------------

    print("TABLE 2B - WEAK SCALING")
    print("-" * 85)
    print(
        f"{'p':>4} {'N total':>12} {'Median ms':>15} "
        f"{'IQR ms':>12} {'Efficiency':>13} "
        f"{'Alpha':>12} {'Gustafson':>12}"
    )

    weak_times = {}
    weak_stage_a = {}

    for p in P_VALUES:
        n = p * N0_WEAK

        median, iqr, stage_a = benchmark_configuration(n, p)

        weak_times[p] = median
        weak_stage_a[p] = stage_a

    t1_weak = weak_times[1]
    gustafson_speedups = {}

    for p in P_VALUES:
        n = p * N0_WEAK
        tp = weak_times[p]

        efficiency = t1_weak / tp
        alpha = weak_stage_a[p] / tp
        gustafson = p - alpha * (p - 1)

        gustafson_speedups[p] = gustafson

        print(
            f"{p:>4} {n:>12,} {tp:>15.4f} "
            f"{'':>12} {efficiency:>13.4f} "
            f"{alpha:>12.6f} {gustafson:>12.4f}"
        )

    print()

    # --------------------------------------------------
    # VERIFICATION HASH
    # --------------------------------------------------

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_SEED,
            strong_speedups[P_LOG],
            gustafson_speedups[P_LOG]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()


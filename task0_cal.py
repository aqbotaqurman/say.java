
import multiprocessing as mp
import time

from harness import emit_verify_tag


# Student information
STUDENT_SEED = 230103276

# Number of clock reads
READS = 2_000_000


def main():
    # Get available CPU count
    cpu_count = mp.cpu_count()

    # Measure clock latency
    start = time.perf_counter_ns()

    for _ in range(READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / READS

    # Print calibration results
    print("PHASE 0 - CALIBRATION RESULTS")
    print("--------------------------------")
    print(f"Student ID: {STUDENT_SEED}")
    print(f"CPU count: {cpu_count}")
    print(f"Clock reads: {READS}")
    print(f"Total time: {total_ns / 1e9:.6f} seconds")
    print(f"Average clock latency: {average_ns:.4f} ns")

    # Generate verification hash
    print()
    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_SEED,
            average_ns,
            float(cpu_count)
        )
    )


if __name__ == "__main__":
    main()


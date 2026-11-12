
import multiprocessing as mp
import time

STUDENT_ID = "230103276"
N = 1_000_000


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    # Measure one million consecutive timer calls
    start = time.perf_counter_ns()

    for _ in range(N):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / N
    total_ms = total_ns / 1_000_000

    logical_cores = mp.cpu_count()

    print("===== PHASE 0: TIMER CALIBRATION =====")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {logical_cores}")
    print(f"Total time: {total_ms:.4f} ms")
    print(f"Average timer latency: {average_ns:.4f} ns")

    # m1 = average latency, m2 = total measurement time
    signature_input = (
        f"PHASE0|{STUDENT_ID}|{logical_cores}|"
        f"{average_ns:.4f}|{total_ms:.4f}"
    )

    signature = fnv1a_hash(signature_input)

    print(
        f"VERIFY_HASH [PHASE0]: {signature:016X} "
        f"(P_log ={logical_cores})"
    )


if __name__ == "__main__":
    main()


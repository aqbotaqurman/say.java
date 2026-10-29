# Karp–Flatt Lab Results

## Hardware profile
- Operating system: [mac]
- Python version: [Python 3.14.8]
- Logical CPU cores: [8]

## Experimental results

| Experiment | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural diagnosis |
|---|---:|---:|---:|---:|---:|---|
| A: Serial Tail | [4.043] | [1.953] | [1.369] | [-0.034] | [0.118] | Non-constant e; fixed serial section not conclusively confirmed |
| B: Lock Contention | [4.368] | [7.977] | [8.515] | [2.653] | [2.266] | Lock contention overhead; parallel execution is slower|
| C: Privatized State | [0.479] | [0.205] | [0.156] | [-0.143] | [0.101] | Improved scaling after privatization; e(4) remains above target |

## Interpretation
### Experiment A: Serial Tail
- **e(2) = -0.034, e(4) = 0.118**
- The values are not similar. Therefore, the results do not fully confirm the expected fixed-serial-section diagnosis. Additional measurements may be needed.

### Experiment B: Lock Contention
- **T4 = 8.515 s, T1 = 4.368 s**
- **e(2) = 2.653, e(4) = 2.266**
- Since T4 > T1, using four processes makes the program slower. However, e(4) is not larger than e(2), so the metric does not show the expected increase.

### Experiment C: Privatization Fix
- **T4 = 0.156 s, T1 = 0.479 s**
- **E4 = (T1 / T4) / 4 × 100% = (0.479 / 0.156) / 4 × 100% = 76.8%**
- Since T4 < T1 and E4 > 70%, the runtime and efficiency criteria are satisfied. However, e(4) = 0.101, which is above the expected threshold of 0.03.


## Notes
Runtimes vary by CPU, operating system, process startup overhead, and background load. Record actual measurements from your machine.

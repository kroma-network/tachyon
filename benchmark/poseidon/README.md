# Poseidon Hash Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)

Run on Apple M3 Pro (12 X 4050 MHz)
CPU Caches:
  L1 Data 64 KiB (x12)
  L1 Instruction 128 KiB (x12)
  L2 Unified 4096 KiB (x12)
```

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark -- --check_results
```

## On Intel i9-13900K

| Repetition | Tachyon      | Arkworks  |
| :--------: | ------------ | --------- |
|     0      | **9.4e-05**  | 0.000123  |
|     1      | **9.1e-05**  | 0.000119  |
|     2      | **9.3e-05**  | 0.000117  |
|     3      | **9.2e-05**  | 0.000117  |
|     4      | **9.2e-05**  | 0.000117  |
|     5      | **9.2e-05**  | 0.00012   |
|     6      | **9.2e-05**  | 0.000118  |
|     7      | **9.2e-05**  | 0.000118  |
|     8      | **9.3e-05**  | 0.000117  |
|     9      | **9.2e-05**  | 0.000116  |
|    avg     | **9.23e-05** | 0.0001182 |

![image](/benchmark/poseidon/poseidon_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

| Repetition | Tachyon      | Arkworks  |
| :--------: | ------------ | --------- |
|     0      | **0.000106** | 0.000128  |
|     1      | **0.0001**   | 0.000121  |
|     2      | **9.7e-05**  | 0.00012   |
|     3      | **9.5e-05**  | 0.000118  |
|     4      | **9.8e-05**  | 0.00012   |
|     5      | **9.8e-05**  | 0.000117  |
|     6      | **9.9e-05**  | 0.000118  |
|     7      | **9.5e-05**  | 0.000116  |
|     8      | **9.7e-05**  | 0.000118  |
|     9      | **9.6e-05**  | 0.000116  |
|    avg     | **9.81e-05** | 0.0001192 |

![image](/benchmark/poseidon/poseidon_benchmark_mac_m3.png)

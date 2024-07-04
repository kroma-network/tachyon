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
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark
```

## On Intel i9-13900K

| Repetition | Tachyon      | Arkworks  |
| :--------: | ------------ | --------- |
|     0      | **9.9e-05**  | 0.000122  |
|     1      | **9.9e-05**  | 0.000119  |
|     2      | **9.9e-05**  | 0.000117  |
|     3      | **9.9e-05**  | 0.000119  |
|     4      | **9.9e-05**  | 0.000118  |
|     5      | **0.000101** | 0.000116  |
|     6      | **9.8e-05**  | 0.000118  |
|     7      | **9.9e-05**  | 0.000118  |
|     8      | **9.9e-05**  | 0.000118  |
|     9      | **9.9e-05**  | 0.000118  |
|    avg     | **9.91e-05** | 0.0001183 |

![image](/benchmark/poseidon/poseidon_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

| Repetition | Tachyon       | Arkworks  |
| :--------: | ------------- | --------- |
|     0      | **0.000112**  | 0.000131  |
|     1      | **0.000118**  | 0.000125  |
|     2      | **0.000111**  | 0.000121  |
|     3      | **0.000103**  | 0.000121  |
|     4      | **0.000104**  | 0.00012   |
|     5      | **0.000103**  | 0.000117  |
|     6      | **0.000103**  | 0.000118  |
|     7      | **0.000104**  | 0.000117  |
|     8      | **0.0001**    | 0.000118  |
|     9      | **0.000113**  | 0.000118  |
|    avg     | **0.0001071** | 0.0001206 |

![image](/benchmark/poseidon/poseidon_benchmark_mac_m3.png)

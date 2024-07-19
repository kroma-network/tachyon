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
|     0      | **3.5e-05**  | 0.000122  |
|     1      | **3.5e-05**  | 0.00012   |
|     2      | **3.6e-05**  | 0.000124  |
|     3      | **3.6e-05**  | 0.000126  |
|     4      | **3.4e-05**  | 0.000123  |
|     5      | **3.4e-05**  | 0.00012   |
|     6      | **3.4e-05**  | 0.00012   |
|     7      | **3.3e-05**  | 0.000118  |
|     8      | **3.4e-05**  | 0.000121  |
|     9      | **3.3e-05**  | 0.00012   |
|    avg     | **3.44e-05** | 0.0001214 |

![image](/benchmark/poseidon/poseidon_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

| Repetition | Tachyon      | Arkworks  |
| :--------: | ------------ | --------- |
|     0      | **4.2e-05**  | 0.000141  |
|     1      | **4.2e-05**  | 0.000143  |
|     2      | **4.2e-05**  | 0.000134  |
|     3      | **4.1e-05**  | 0.00014   |
|     4      | **4.1e-05**  | 0.000131  |
|     5      | **4.1e-05**  | 0.000128  |
|     6      | **4.1e-05**  | 0.000127  |
|     7      | **4.2e-05**  | 0.00013   |
|     8      | **4.1e-05**  | 0.000128  |
|     9      | **4.1e-05**  | 0.000132  |
|    avg     | **4.14e-05** | 0.0001334 |

![image](/benchmark/poseidon/poseidon_benchmark_mac_m3.png)

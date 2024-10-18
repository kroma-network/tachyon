# Poseidon Hash Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
Compiler: clang-15
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
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark -- --check_results
```

## On Intel i9-13900K

| Repetition | Tachyon     | Arkworks |
| :--------: | ----------- | -------- |
|     0      | **3.5e-05** | 0.000107 |
|     1      | **3.2e-05** | 0.000106 |
|     2      | **3.2e-05** | 0.000108 |
|     3      | **3.1e-05** | 0.000107 |
|     4      | **3.1e-05** | 0.000107 |
|     5      | **3.1e-05** | 0.000107 |
|     6      | **3.1e-05** | 0.000104 |
|     7      | **3.1e-05** | 0.000105 |
|     8      | **3.1e-05** | 0.000106 |
|     9      | **3.1e-05** | 0.000107 |
|    avg     | **3.1e-05** | 0.000106 |

![image](/benchmark/poseidon/poseidon_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

| Repetition | Tachyon     | Arkworks |
| :--------: | ----------- | -------- |
|     0      | **3.7e-05** | 0.000111 |
|     1      | **3.5e-05** | 0.000105 |
|     2      | **3.3e-05** | 0.000103 |
|     3      | **3.2e-05** | 0.000104 |
|     4      | **3.2e-05** | 0.000101 |
|     5      | **3.2e-05** | 0.000103 |
|     6      | **3.2e-05** | 0.000105 |
|     7      | **3.2e-05** | 0.000102 |
|     8      | **3.2e-05** | 0.000102 |
|     9      | **3.2e-05** | 0.000102 |
|    avg     | **3.2e-05** | 0.000103 |

![image](/benchmark/poseidon/poseidon_benchmark_mac_m3.png)

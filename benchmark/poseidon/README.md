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
bazel run --config opt --//:has_rtti --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark -- --check_results
```

## On Intel i9-13900K

| Repetition | Tachyon     | Arkworks |
| :--------: | ----------- | -------- |
|     0      | **3.4e-05** | 0.000103 |
|     1      | **3.4e-05** | 0.000103 |
|     2      | **3.3e-05** | 0.000101 |
|     3      | **3.3e-05** | 0.000103 |
|     4      | **3.3e-05** | 0.000103 |
|     5      | **3.3e-05** | 0.000108 |
|     6      | **3.3e-05** | 0.000102 |
|     7      | **3.2e-05** | 0.000104 |
|     8      | **3.3e-05** | 0.000102 |
|     9      | **3.1e-05** | 0.000103 |
|    avg     | **3.2e-05** | 0.000103 |

![image](/benchmark/poseidon/poseidon_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

| Repetition | Tachyon     | Arkworks |
| :--------: | ----------- | -------- |
|     1      | **3.9e-05** | 0.000108 |
|     0      | **3.8e-05** | 0.000111 |
|     2      | **3.7e-05** | 0.000106 |
|     3      | **3.6e-05** | 0.000104 |
|     4      | **3.6e-05** | 0.000106 |
|     5      | **3.5e-05** | 0.000103 |
|     6      | **3.5e-05** | 0.000103 |
|     7      | **3.5e-05** | 0.000103 |
|     8      | **3.5e-05** | 0.000103 |
|     9      | **3.5e-05** | 0.000105 |
|    avg     | **3.6e-05** | 0.000105 |

![image](/benchmark/poseidon/poseidon_benchmark_mac_m3.png)

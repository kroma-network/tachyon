# Poseidon Hash Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
```

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark
```

| Repetition | Tachyon  | Arkworks      |
| :--------: | -------- | ------------- |
|     0      | 0.000124 | **0.000114**  |
|     1      | 0.000125 | **0.00011**   |
|     2      | 0.000124 | **0.00011**   |
|     3      | 0.000124 | **0.000106**  |
|     4      | 0.000124 | **0.00011**   |
|     5      | 0.000124 | **0.000109**  |
|     6      | 0.000125 | **0.000108**  |
|     7      | 0.000123 | **0.00011**   |
|     8      | 0.000123 | **0.000109**  |
|     9      | 0.000124 | **0.000106**  |
|    avg     | 0.000124 | **0.0001092** |

![image](/benchmark/poseidon/Poseidon%20Benchmark.png)

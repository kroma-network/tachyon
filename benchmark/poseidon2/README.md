# Poseidon2 Hash Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
```

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p bn254_fr --vendor horizen --vendor plonky3
```

| Repetition | tachyon  | horizen | plonky3 |
| :--------- | -------- | ------- | ------- |
| 0          | 1.2e-05  | 7e-06   | 1e-05   |
| 1          | 1.1e-05  | 4e-06   | 8e-06   |
| 2          | 1.1e-05  | 4e-06   | 8e-06   |
| 3          | 1.1e-05  | 3e-06   | 8e-06   |
| 4          | 1e-05    | 3e-06   | 7e-06   |
| 5          | 1.1e-05  | 3e-06   | 7e-06   |
| 6          | 1e-05    | 3e-06   | 7e-06   |
| 7          | 1e-05    | 3e-06   | 7e-06   |
| 8          | 1.1e-05  | 3e-06   | 7e-06   |
| 9          | 1e-05    | 3e-06   | 7e-06   |
| avg        | 1.07e-05 | 3.6e-06 | 7.6e-06 |

![image](/benchmark/poseidon2/Poseidon2%20Benchmark.png)

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
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon:poseidon_benchmark -- -s 10000 -a 10000
```

| Repetition | Tachyon  | Arkworks     |
| :--------: | -------- | ------------ |
|     0      | 0.365703 | **0.26695**  |
|     1      | 0.383136 | **0.267033** |
|     2      | 0.383506 | **0.272285** |
|     3      | 0.374358 | **0.257264** |
|     4      | 0.370592 | **0.25442**  |
|     5      | 0.371024 | **0.2545**   |
|     6      | 0.373796 | **0.263127** |
|     7      | 0.375338 | **0.27176**  |
|     8      | 0.374944 | **0.267881** |
|     9      | 0.374335 | **0.262994** |
|    avg     | 0.374673 | **0.263821** |

![image](/benchmark/poseidon/Poseidon%20Benchmark.png)

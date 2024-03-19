# MSM Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
```

## Random points with bellman msm algorithm

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2
```

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.037107** | 0.043005 | 0.079576 | 0.04854  |
|    17    | **0.051857** | 0.078176 | 0.11077  | 0.087005 |
|    18    | **0.10235**  | 0.15416  | 0.179148 | 0.146375 |
|    19    | **0.185314** | 0.301129 | 0.360499 | 0.273742 |
|    20    | **0.352276** | 0.592931 | 0.516447 | 0.533842 |
|    21    | **0.630413** | 1.07979  | 1.00715  | 1.01114  |
|    22    | **1.22393**  | 2.18742  | 2.00827  | 1.99418  |
|    23    | **2.34737**  | 4.34471  | 3.2081   | 3.62107  |

![image](/benchmark/msm/MSM%20Benchmark(random,%20bellman_msm).png)

## Non-uniform points with bellman msm algorithm

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --test_set non_uniform
```

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | 0.041756     | **0.038514** | 0.050397 | 0.040222 |
|    17    | **0.06304**  | 0.069963     | 0.067314 | 0.07276  |
|    18    | **0.09546**  | 0.137724     | 0.126665 | 0.138729 |
|    19    | **0.189892** | 0.261193     | 0.262439 | 0.26166  |
|    20    | **0.313575** | 0.475754     | 0.387448 | 0.51257  |
|    21    | **0.550828** | 0.901562     | 0.666334 | 0.955411 |
|    22    | **1.14408**  | 1.61699      | 1.29853  | 1.88522  |
|    23    | **1.91659**  | 3.13911      | 2.16368  | 3.46701  |

![image](/benchmark/msm/MSM%20Benchmark(non_uniform,%20bellman_msm).png)

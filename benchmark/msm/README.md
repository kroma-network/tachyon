# MSM Benchmark

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

## Random points with bellman msm algorithm

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2
```

### On Intel i9-13900K

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

![image](</benchmark/msm/MSM%20Benchmark(random,%20bellman_msm).png>)

### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.049285** | 0.052368 | 0.118564 | 0.079168 |
|    17    | **0.088628** | 0.105875 | 0.16704  | 0.142758 |
|    18    | **0.157609** | 0.19186  | 0.299953 | 0.278592 |
|    19    | **0.282686** | 0.351326 | 0.578682 | 0.506371 |
|    20    | **0.571241** | 0.702036 | 0.901252 | 0.974515 |
|    21    | **1.106550** | 1.54553  | 1.63521  | 1.85615  |
|    22    | **2.276600** | 3.35888  | 3.274    | 3.68391  |
|    23    | **4.191330** | 6.41272  | 5.86292  | 6.89936  |

![image](</benchmark/msm/MSM%20Benchmark%20MacM3(random,%20bellman_msm).png>)

## Non-uniform points with bellman msm algorithm

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --test_set non_uniform
```

### On Intel i9-13900K

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

![image](</benchmark/msm/MSM%20Benchmark(non_uniform,%20bellman_msm).png>)

### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | **0.040045** | 0.049862     | 0.083292 | 0.082423 |
|    17    | **0.074318** | 0.091205     | 0.128679 | 0.157111 |
|    18    | **0.140125** | 0.177842     | 0.232212 | 0.310648 |
|    19    | **0.287691** | 0.330268     | 0.437016 | 0.535915 |
|    20    | **0.55437**  | 0.651841     | 0.713282 | 1.01238  |
|    21    | **1.01053**  | 1.36348      | 1.29945  | 1.75816  |
|    22    | **2.00677**  | 2.56         | 2.49532  | 3.55769  |
|    23    | **4.02119**  | 5.2982       | 4.56454  | 7.11582  |

![image](</benchmark/msm/MSM%20Benchmark%20MacM3(non_uniform,%20bellman_msm).png>)

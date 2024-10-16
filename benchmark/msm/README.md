# MSM Benchmark

## CPU

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

Note: Run with `build --@rules_rust//:extra_rustc_flags="-Ctarget-cpu=native"` in your .bazelrc.user

### Uniform points

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.028461** | 0.037741 | 0.077416 | 0.045742 |
|    17    | **0.059648** | 0.074936 | 0.105104 | 0.08211  |
|    18    | **0.08743**  | 0.12735  | 0.196602 | 0.151715 |
|    19    | **0.181646** | 0.252424 | 0.319185 | 0.282056 |
|    20    | **0.303829** | 0.454595 | 0.471094 | 0.526231 |
|    21    | **0.549287** | 0.951397 | 0.886244 | 1.00624  |
|    22    | **1.11021**  | 2.00783  | 1.72011  | 1.9662   |
|    23    | **2.06762**  | 3.78478  | 2.76673  | 3.68139  |

![image](/benchmark/msm/msm_benchmark_uniform_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.046099** | 0.051773 | 0.110882 | 0.09505  |
|    17    | **0.079298** | 0.097698 | 0.166183 | 0.174984 |
|    18    | **0.151962** | 0.173607 | 0.296879 | 0.337657 |
|    19    | **0.287848** | 0.34129  | 0.5563   | 0.592885 |
|    20    | **0.504987** | 0.630489 | 0.840907 | 1.07097  |
|    21    | **0.980302** | 1.33391  | 1.56196  | 1.98335  |
|    22    | **1.89977**  | 2.86768  | 3.04392  | 3.9341   |
|    23    | **3.73015**  | 5.71419  | 5.45636  | 7.51033  |

![image](/benchmark/msm/msm_benchmark_uniform_mac_m3.png)

### Non-uniform points

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/msm:msm_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --test_set non_uniform --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.030188** | 0.033608 | 0.057795 | 0.060642 |
|    17    | **0.048851** | 0.064059 | 0.132584 | 0.099568 |
|    18    | **0.080146** | 0.121525 | 0.124192 | 0.147496 |
|    19    | **0.147626** | 0.227517 | 0.234429 | 0.27793  |
|    20    | **0.289661** | 0.445139 | 0.341189 | 0.509375 |
|    21    | **0.495707** | 0.801975 | 0.702259 | 1.0386   |
|    22    | **0.993738** | 1.51266  | 1.24812  | 1.88462  |
|    23    | **1.69944**  | 3.07904  | 2.00071  | 3.57452  |

![image](/benchmark/msm/msm_benchmark_non_uniform_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.040954** | 0.046663 | 0.076068 | 0.089352 |
|    17    | **0.069956** | 0.089339 | 0.119363 | 0.166812 |
|    18    | **0.146869** | 0.163578 | 0.225768 | 0.326553 |
|    19    | **0.268475** | 0.302439 | 0.460063 | 0.579915 |
|    20    | **0.501956** | 0.627272 | 0.723071 | 1.09316  |
|    21    | **0.920728** | 1.20662  | 1.22352  | 1.98457  |
|    22    | **1.78902**  | 2.40124  | 2.24543  | 3.83765  |
|    23    | **3.47906**  | 4.70651  | 4.13381  | 7.43978  |

![image](/benchmark/msm/msm_benchmark_non_uniform_mac_m3.png)

## GPU

### Uniform points

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib --config cuda //benchmark/msm:msm_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --test_set non_uniform --check_results
```

#### On RTX-4090

| Exponent | Tachyon CPU | Tachyon GPU  |
| :------: | ----------- | ------------ |
|    16    | 0.026688    | **0.01981**  |
|    17    | 0.041291    | **0.006624** |
|    18    | 0.081467    | **0.008306** |
|    19    | 0.148929    | **0.012553** |
|    20    | 0.260831    | **0.02423**  |
|    21    | 0.474542    | **0.044591** |
|    22    | 0.921276    | **0.088349** |
|    23    | 1.70264     | **0.162646** |

![image](/benchmark/msm/msm_benchmark_uniform_ubuntu_rtx_4090.png)

### Non-uniform points

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib --config cuda //benchmark/msm:msm_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --check_results
```

#### On RTX-4090

| Exponent | Tachyon CPU | Tachyon GPU  |
| :------: | ----------- | ------------ |
|    16    | 0.029045    | **0.020228** |
|    17    | 0.047588    | **0.006565** |
|    18    | 0.089673    | **0.008864** |
|    19    | 0.164875    | **0.012308** |
|    20    | 0.29135     | **0.023396** |
|    21    | 0.541067    | **0.043512** |
|    22    | 1.0379      | **0.08407**  |
|    23    | 2.08601     | **0.157046** |

![image](/benchmark/msm/msm_benchmark_non_uniform_ubuntu_rtx_4090.png)

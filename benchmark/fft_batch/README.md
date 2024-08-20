# FFTBatch/CosetLDEBatch Benchmark

## CPU

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

### FFTBatch

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor plonky3 -p baby_bear --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 16       | **0.003543** | 0.005809     |
| 17       | **0.007168** | 0.009239     |
| 18       | 0.027791     | **0.023848** |
| 19       | 0.063468     | **0.049085** |
| 20       | 0.133178     | **0.102343** |
| 21       | 0.238817     | **0.208557** |
| 22       | 0.507061     | **0.427260** |
| 23       | 1.11136      | **0.922439** |

![image](/benchmark/fft_batch/fft_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Plonky3  |
| :------- | ------------ | -------- |
| 16       | **0.007926** | 0.01639  |
| 17       | **0.016391** | 0.024321 |
| 18       | **0.035098** | 0.085961 |
| 19       | **0.076266** | 0.096928 |
| 20       | **0.145975** | 0.151024 |
| 21       | **0.309752** | 0.339549 |
| 22       | **0.674991** | 2.66605  |
| 23       | **1.727520** | 7.79002  |

![image](/benchmark/fft_batch/fft_batch_benchmark_mac_m3.png)

### CosetLDEBatch

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor plonky3 -p baby_bear --run_coset_lde --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Plonky3  |
| :------- | ------------ | -------- |
| 16       | **0.008384** | 0.018529 |
| 17       | **0.017164** | 0.043266 |
| 18       | **0.052999** | 0.093348 |
| 19       | **0.128624** | 0.19531  |
| 20       | **0.246412** | 0.418079 |
| 21       | **0.508587** | 0.816136 |
| 22       | **1.071360** | 1.63289  |
| 23       | **2.225130** | 3.53348  |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 16       | **0.013503** | 0.020251     |
| 17       | **0.028850** | 0.044731     |
| 18       | 0.061675     | **0.055980** |
| 19       | 0.132670     | **0.113062** |
| 20       | **0.263582** | 0.300933     |
| 21       | **0.548267** | 0.573641     |
| 22       | **1.164400** | 1.19051      |
| 23       | **2.818220** | 5.11368      |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_mac_m3.png)

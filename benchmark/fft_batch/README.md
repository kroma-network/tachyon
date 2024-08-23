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
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 -k 26 --vendor plonky3 -p baby_bear --check_results
```

WARNING: On Mac M3, tests beyond degree 24 are not feasible due to memory constraints.

#### On Intel i9-13900K

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 20       | 0.117925     | **0.110098** |
| 21       | 0.222959     | **0.218505** |
| 22       | 0.459209     | **0.447758** |
| 23       | 0.97874      | **0.964644** |
| 24       | 2.09675      | **2.092210** |
| 25       | **6.20441**  | 6.98453      |
| 26       | **18.6084**  | 20.7476      |

![image](/benchmark/fft_batch/fft_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon   | Plonky3      |
| :------- | --------- | ------------ |
| 20       | 0.132521  | **0.072505** |
| 21       | 0.287744  | **0.140527** |
| 22       | 0.588894  | **0.280177** |
| 23       | 1.17446   | **0.621024** |
| 24       | 3.17213   | **2.399220** |

![image](/benchmark/fft_batch/fft_batch_benchmark_mac_m3.png)

### CosetLDEBatch

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 -k 26 --vendor plonky3 -p baby_bear --run_coset_lde --check_results
```

WARNING: On Mac M3, tests beyond degree 24 are not feasible due to memory constraints.

#### On Intel i9-13900K

| Exponent | Tachyon      | Plonky3  |
| :------- | ------------ | -------- |
| 20       | **0.241410** | 0.396681 |
| 21       | **0.480885** | 0.794424 |
| 22       | **0.978230** | 1.60685  |
| 23       | **2.005920** | 3.48347  |
| 24       | **4.615940** | 7.89591  |
| 25       | **12.62420** | 22.158   |
| 26       | **35.45570** | 56.9609  |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon   | Plonky3      |
| :------- | --------- | ------------ |
| 20       | 0.269538  | **0.204846** |
| 21       | 0.543247  | **0.414618** |
| 22       | 1.15925   | **0.877114** |
| 23       | 2.43017   | **1.835210** |
| 24       | 6.89016   | **4.846630** |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_mac_m3.png)

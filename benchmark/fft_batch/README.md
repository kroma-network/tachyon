# FFTBatch/CosetLDEBatch Benchmark

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

### FFTBatch

WARNING: On Mac M3, tests beyond degree 24 are not feasible due to memory constraints.

#### On Intel i9-13900K

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 -k 26 --vendor plonky3 -p baby_bear --check_results
```

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 20       | **0.092595** | 0.094762     |
| 21       | **0.191168** | 0.193567     |
| 22       | 0.406239     | **0.384377** |
| 23       | 0.892501     | **0.842694** |
| 24       | 1.91177      | **1.90586**  |
| 25       | **5.82862**  | 7.34128      |
| 26       | **17.1807**  | 20.3968      |

![image](/benchmark/fft_batch/fft_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24  --vendor plonky3 -p baby_bear --check_results
```

| Exponent | Tachyon  | Plonky3      |
| :------- | -------- | ------------ |
| 20       | 0.083416 | **0.066952** |
| 21       | 0.194191 | **0.138168** |
| 22       | 0.408045 | **0.299547** |
| 23       | 0.955439 | **0.679252** |
| 24       | 11.8495  | **6.47188**  |

![image](/benchmark/fft_batch/fft_batch_benchmark_mac_m3.png)

### CosetLDEBatch

WARNING: On Intel i9-13900K, tests beyond degree 25 are not feasible due to memory constraints, and on Mac M3, tests beyond degree 24 are not feasible due to memory constraints.

#### On Intel i9-13900K

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark --@rules_rust//:extra_rustc_flag="--cfg=feature=\"parallel\"" -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 --vendor plonky3 -p baby_bear --run_coset_lde --check_results
```

| Exponent | Tachyon     | Plonky3  |
| :------- | ----------- | -------- |
| 20       | **0.46917** | 0.639744 |
| 21       | **0.92528** | 1.2923   |
| 22       | **1.87363** | 2.68427  |
| 23       | **4.06008** | 5.67987  |
| 24       | **9.6627**  | 14.6164  |
| 25       | **25.7953** | 39.5498  |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 --vendor plonky3 -p baby_bear --run_coset_lde --check_results
```

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 20       | **0.318485** | 0.323865     |
| 21       | 0.667106     | **0.660975** |
| 22       | **1.44873**  | 3.40795      |
| 23       | 8.27201      | **5.91238**  |
| 24       | 39.9678      | **23.1033**  |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_mac_m3.png)

# (I)FFT Benchmark

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

### FFT

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | **0.002058** | 0.005143     | 0.006314 | 0.002249 |
|    17    | **0.002246** | 0.00334      | 0.015646 | 0.006193 |
|    18    | **0.010154** | 0.018807     | 0.046443 | 0.007574 |
|    19    | 0.022984     | **0.014652** | 0.076281 | 0.014506 |
|    20    | **0.02**     | 0.02497      | 0.100082 | 0.042877 |
|    21    | **0.044831** | 0.075563     | 0.20222  | 0.067161 |
|    22    | **0.130201** | 0.179075     | 0.402452 | 0.169194 |
|    23    | **0.281398** | 0.394068     | 0.792004 | 0.372566 |

![image](/benchmark/fft/fft_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.002526** | 0.003804 | 0.00784  | 0.005689 |
|    17    | **0.004694** | 0.005769 | 0.015577 | 0.01121  |
|    18    | **0.009246** | 0.010243 | 0.027834 | 0.022379 |
|    19    | **0.018328** | 0.020404 | 0.055661 | 0.041394 |
|    20    | **0.039683** | 0.041085 | 0.110702 | 0.086299 |
|    21    | **0.079138** | 0.087336 | 0.230857 | 0.175599 |
|    22    | **0.166646** | 0.177959 | 0.474296 | 0.352872 |
|    23    | **0.33996**  | 0.363612 | 0.971581 | 0.748284 |

![image](/benchmark/fft/fft_benchmark_mac_m3.png)

### IFFT

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --run_ifft --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2        |
| :------: | ------------ | -------- | -------- | ------------ |
|    16    | **0.001392** | 0.012028 | 0.009913 | 0.002413     |
|    17    | **0.002511** | 0.00427  | 0.01418  | 0.005731     |
|    18    | 0.01762      | 0.021167 | 0.034676 | **0.010811** |
|    19    | **0.009646** | 0.01447  | 0.058714 | 0.016038     |
|    20    | **0.030303** | 0.034815 | 0.104936 | 0.05337      |
|    21    | **0.047463** | 0.072579 | 0.199788 | 0.093146     |
|    22    | **0.146697** | 0.181389 | 0.391296 | 0.19874      |
|    23    | **0.285937** | 0.403596 | 0.82276  | 0.347876     |

![image](/benchmark/fft/ifft_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.002798** | 0.003867 | 0.008102 | 0.005665 |
|    17    | **0.004882** | 0.005737 | 0.015998 | 0.011672 |
|    18    | **0.010308** | 0.010962 | 0.028118 | 0.022723 |
|    19    | **0.018724** | 0.021338 | 0.056855 | 0.042554 |
|    20    | **0.037687** | 0.043237 | 0.113848 | 0.089899 |
|    21    | **0.078429** | 0.092134 | 0.234585 | 0.174939 |
|    22    | **0.162542** | 0.189442 | 0.484644 | 0.361127 |
|    23    | **0.338646** | 0.392674 | 0.989173 | 0.765592 |

![image](/benchmark/fft/ifft_benchmark_mac_m3.png)

## GPU

### FFT

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --config cuda --//:has_matplotlib //benchmark/fft:fft_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --check_results
```

#### On RTX-4090

| Exponent | Tachyon CPU | Tachyon GPU  |
| :------: | ----------- | ------------ |
|    16    | 0.002348    | **0.001**    |
|    17    | 0.00204     | **0.001182** |
|    18    | 0.00393     | **0.002211** |
|    19    | 0.009317    | **0.004079** |
|    20    | 0.049204    | **0.008114** |
|    21    | 0.044158    | **0.01616**  |
|    22    | 0.134064    | **0.032785** |
|    23    | 0.274101    | **0.066068** |

![image](/benchmark/fft/fft_benchmark_ubuntu_rtx_4090.png)

### IFFT

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --config cuda --//:has_matplotlib //benchmark/fft:fft_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --run_ifft --check_results
```

#### On RTX-4090

| Exponent | Tachyon  | Tachyon GPU  |
| :------: | -------- | ------------ |
|    16    | 0.002138 | **0.001341** |
|    17    | 0.00488  | **0.000933** |
|    18    | 0.003887 | **0.002502** |
|    19    | 0.00896  | **0.003806** |
|    20    | 0.017953 | **0.007745** |
|    21    | 0.043787 | **0.016268** |
|    22    | 0.132048 | **0.033012** |
|    23    | 0.291132 | **0.066022** |

![image](/benchmark/fft/ifft_benchmark_ubuntu_rtx_4090.png)

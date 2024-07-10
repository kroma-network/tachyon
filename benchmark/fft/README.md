# (I)FFT Benchmark

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

### FFT

```shell
bazel run --config halo2 -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | **0.000958** | 0.004086     | 0.007342 | 0.003784 |
|    17    | 0.032529     | **0.003283** | 0.012624 | 0.005433 |
|    18    | 0.014067     | **0.005768** | 0.025811 | 0.009372 |
|    19    | **0.008459** | 0.011465     | 0.05208  | 0.019333 |
|    20    | **0.016166** | 0.024533     | 0.106217 | 0.042381 |
|    21    | **0.039447** | 0.069444     | 0.212414 | 0.087621 |
|    22    | **0.125954** | 0.177245     | 0.431237 | 0.188843 |
|    23    | **0.297259** | 0.391987     | 0.835686 | 0.427426 |

![image](/benchmark/fft/fft_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | **0.002735** | 0.003468     | 0.007731 | 0.006372 |
|    17    | **0.005237** | 0.006043     | 0.015891 | 0.012804 |
|    18    | **0.009494** | 0.010686     | 0.027312 | 0.02485  |
|    19    | 0.020251     | **0.020156** | 0.055652 | 0.045714 |
|    20    | **0.038186** | 0.040006     | 0.110531 | 0.096778 |
|    21    | **0.085204** | 0.087181     | 0.228044 | 0.191695 |
|    22    | **0.166863** | 0.179635     | 0.472941 | 0.386844 |
|    23    | **0.347128** | 0.378249     | 0.970552 | 0.814043 |

![image](/benchmark/fft/fft_benchmark_mac_m3.png)

### IFFT

```shell
bazel run -c opt --config halo2 --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --run_ifft --check_results
```

#### On Intel i9-13900K

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2       |
| :------: | ------------ | ------------ | -------- | ----------- |
|    16    | 0.003078     | 0.004531     | 0.007794 | **0.00297** |
|    17    | 0.011666     | **0.005005** | 0.012804 | 0.005309    |
|    18    | **0.005614** | 0.009204     | 0.025717 | 0.009741    |
|    19    | **0.007625** | 0.015332     | 0.050253 | 0.018729    |
|    20    | **0.016751** | 0.030142     | 0.111549 | 0.041873    |
|    21    | **0.039565** | 0.0715       | 0.222403 | 0.098125    |
|    22    | **0.140152** | 0.181124     | 0.415709 | 0.188011    |
|    23    | **0.317353** | 0.400472     | 0.845031 | 0.407396    |

![image](/benchmark/fft/ifft_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Arkworks | Bellman  | Halo2    |
| :------: | ------------ | -------- | -------- | -------- |
|    16    | **0.002766** | 0.004274 | 0.007948 | 0.006638 |
|    17    | **0.005883** | 0.006978 | 0.016308 | 0.013121 |
|    18    | **0.010532** | 0.012815 | 0.029066 | 0.028791 |
|    19    | **0.020781** | 0.024054 | 0.059351 | 0.048824 |
|    20    | **0.041061** | 0.048806 | 0.11825  | 0.099004 |
|    21    | **0.090855** | 0.101232 | 0.236775 | 0.210805 |
|    22    | **0.170776** | 0.203109 | 0.488306 | 0.423618 |
|    23    | **0.383255** | 0.454968 | 1.03129  | 0.881795 |

![image](/benchmark/fft/ifft_benchmark_mac_m3.png)

## GPU

### FFT

```shell
bazel run -c opt --config cuda --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --check_results
```

#### On RTX-4090

| Exponent | Tachyon     | Tachyon GPU  |
| :------: | ----------- | ------------ |
|    16    | **0.00097** | 0.001231     |
|    17    | 0.002156    | **0.000667** |
|    18    | 0.003524    | **0.001297** |
|    19    | 0.007366    | **0.002654** |
|    20    | 0.015787    | **0.005877** |
|    21    | 0.03753     | **0.012573** |
|    22    | 0.122167    | **0.027632** |
|    23    | 0.268875    | **0.055971** |

![image](/benchmark/fft/fft_benchmark_ubuntu_rtx_4090.png)

### IFFT

```shell
bazel run -c opt --config cuda --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark_gpu -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --run_ifft --check_results
```

#### On RTX-4090

| Exponent | Tachyon  | Tachyon GPU  |
| :------: | -------- | ------------ |
|    16    | 0.000993 | **0.000833** |
|    17    | 0.001673 | **0.000643** |
|    18    | 0.003533 | **0.001305** |
|    19    | 0.007446 | **0.002701** |
|    20    | 0.016039 | **0.005882** |
|    21    | 0.03786  | **0.012817** |
|    22    | 0.126032 | **0.027767** |
|    23    | 0.32731  | **0.056064** |

![image](/benchmark/fft/ifft_benchmark_ubuntu_rtx_4090.png)

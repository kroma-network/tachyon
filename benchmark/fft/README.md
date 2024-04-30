# (I)FFT Benchmark

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
```

## FFT

```shell
bazel run --config halo2 -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --vendor arkworks --vendor bellman --vendor halo2 --check_results
```

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

![image](/benchmark/fft/FFT%20Benchmark.png)

## IFFT

```shell
bazel run -c opt --config halo2 --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --run_ifft
```

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

![image](/benchmark/fft/IFFT%20Benchmark.png)

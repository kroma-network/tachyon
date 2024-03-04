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
bazel run -c opt --config halo2 --//:has_openmp --//:polygon_zkevm_backend --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23
```

| Exponent | Tachyon      | Halo2        |
| :------: | ------------ | ------------ |
|    16    | **0.001964** | 0.005087     |
|    17    | **0.006411** | 0.00705      |
|    18    | **0.016552** | 0.023917     |
|    19    | **0.027728** | 0.067646     |
|    20    | 0.057617     | **0.056551** |
|    21    | 0.123259     | **0.086488** |
|    22    | 0.297385     | **0.18532**  |
|    23    | 0.619081     | **0.399886** |

![image](/benchmark/fft/FFT%20Benchmark.png)

## IFFT

```shell
bazel run -c opt --config halo2 --//:has_openmp --//:polygon_zkevm_backend --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --run_ifft
```

| Exponent | Tachyon      | Halo2        |
| :------: | ------------ | ------------ |
|    16    | **0.002298** | 0.004557     |
|    17    | **0.005008** | 0.005661     |
|    18    | **0.009929** | 0.011304     |
|    19    | **0.039518** | 0.044937     |
|    20    | **0.045926** | 0.08374      |
|    21    | 0.130692     | **0.108806** |
|    22    | 0.316693     | **0.204709** |
|    23    | 0.637058     | **0.451356** |

![image](/benchmark/fft/IFFT%20Benchmark.png)

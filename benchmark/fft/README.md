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

| Exponent | Tachyon  | Arkworks     | Bellman  | Halo2        |
| :------: | -------- | ------------ | -------- | ------------ |
|    16    | 0.00211  | 0.005267     | 0.006479 | **0.003764** |
|    17    | 0.012948 | **0.00457**  | 0.01275  | 0.005007     |
|    18    | 0.01599  | **0.007549** | 0.025162 | 0.009437     |
|    19    | 0.027597 | **0.012709** | 0.053993 | 0.021365     |
|    20    | 0.053458 | **0.027794** | 0.109091 | 0.046986     |
|    21    | 0.125161 | **0.07487**  | 0.230635 | 0.101493     |
|    22    | 0.300036 | **0.195025** | 0.464866 | 0.21763      |
|    23    | 0.687611 | **0.417051** | 0.985956 | 0.441969     |

![image](/benchmark/fft/FFT%20Benchmark.png)

## IFFT

```shell
bazel run -c opt --config halo2 --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft:fft_benchmark -- -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 --run_ifft
```

| Exponent | Tachyon      | Arkworks     | Bellman  | Halo2    |
| :------: | ------------ | ------------ | -------- | -------- |
|    16    | **0.002577** | 0.005436     | 0.007644 | 0.003944 |
|    17    | 0.01929      | **0.005533** | 0.013494 | 0.005947 |
|    18    | 0.010847     | **0.009226** | 0.026845 | 0.011326 |
|    19    | 0.022843     | **0.018677** | 0.054128 | 0.022901 |
|    20    | 0.047196     | **0.038015** | 0.111437 | 0.048271 |
|    21    | 0.125933     | **0.084934** | 0.232147 | 0.10333  |
|    22    | 0.323088     | **0.20301**  | 0.469865 | 0.220576 |
|    23    | 0.637929     | **0.446485** | 0.967357 | 0.455753 |

![image](/benchmark/fft/IFFT%20Benchmark.png)

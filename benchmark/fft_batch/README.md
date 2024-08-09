# FFTBatch Benchmark

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

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:benchmark -- -d 21 -p baby_bear --vendor plonky3
```

#### On Intel i9-13900K

| Repetition | Tachyon  |  Plonky3      |
| :--------- | -------  |  -----------  |
| 0          | 1.49794  |  **0.211087** |
| 1          | 1.14541  |  **0.209432** |
| 2          | 1.19469  |  **0.212014** |
| 3          | 1.19882  |  **0.209351** |
| 4          | 1.13655  |  **0.210366** |
| 5          | 0.901086 |  **0.209983** |
| 6          | 1.19522  |  **0.211067** |
| 7          | 1.18459  |  **0.209839** |
| 8          | 0.900512 |  **0.209745** |
| 9          | 1.21875  |  **0.208887** |
| avg        | 1.15736  |  **0.210177** |

![image](/benchmark/fft_batch/benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Repetition | Tachyon | Plonky3      |
| :--------- | ------- | ------------ |
| 0          | 2.53273 | **0.146832** |
| 1          | 2.48872 | **0.158379** |
| 2          | 2.52125 | **0.147676** |
| 3          | 2.53502 | **0.162602** |
| 4          | 2.50802 | **0.144537** |
| 5          | 2.57135 | **0.145066** |
| 6          | 2.4639  | **0.144031** |
| 7          | 2.53492 | **0.150916** |
| 8          | 2.52831 | **0.149961** |
| 9          | 2.53022 | **0.167055** |
| avg        | 2.52144 | **0.151705** |

![image](/benchmark/fft_batch/benchmark_mac_m3.png)

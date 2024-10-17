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
bazel run --config opt --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 -k 26 --vendor plonky3 -p baby_bear --check_results
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
bazel run --config opt --//:has_rtti --//:has_matplotlib //benchmark/fft_batch:fft_batch_benchmark -- -k 20 -k 21 -k 22 -k 23 -k 24 -k 25 --vendor plonky3 -p baby_bear --run_coset_lde --check_results
```

WARNING: On Mac M3, tests beyond degree 24 are not feasible due to memory constraints.

#### On Intel i9-13900K

| Exponent | Tachyon      | Plonky3  |
| :------- | ------------ | -------- |
| 20       | **0.414096** | 0.783275 |
| 21       | **0.828539** | 1.47701  |
| 22       | **1.784080** | 3.06198  |
| 23       | **3.673930** | 6.49181  |
| 24       | **9.325390** | 16.2383  |
| 25       | **25.66560** | 41.3335  |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Exponent | Tachyon      | Plonky3      |
| :------- | ------------ | ------------ |
| 18       | 0.100942     | **0.086087** |
| 19       | 0.214471     | **0.182212** |
| 20       | 0.481229     | **0.359246** |
| 21       | **0.981806** | 1.518190     |
| 22       | 3.86094      | **3.244580** |
| 23       | 7.50879      | **6.052250** |

![image](/benchmark/fft_batch/coset_lde_batch_benchmark_mac_m3.png)

# FRI Benchmark

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

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/fri:fri_benchmark -- -k 18 -k 19 -k 20 -k 21 -k 22 --batch_size 100 --input_num 4 --round_num 4 --log_blowup 2 --vendor plonky3 --check_results
```

## On Intel i9-13900K

| Exponent | Tachyon     | Plonky3 |
| :------- | ----------- | ------- |
| 18       | **1.59124** | 2.36518 |
| 19       | **2.87866** | 4.65791 |
| 20       | **6.06711** | 9.5114  |
| 21       | **12.1177** | 19.0475 |
| 22       | **24.4839** | 38.4716 |

![image](/benchmark/fri/fri_benchmark_ubuntu_i9.png)

## On Mac M3 Pro

WARNING: On Mac M3, high degree tests are not feasible due to memory constraints.

| Exponent | Tachyon | Plonky3 |
| :------- | ------- | ------- |
| 18       | 3.96588 | 2.92354 |
| 19       | 7.95329 | 5.89079 |
| 20       | 15.8636 | 11.8225 |
| 21       | 46.1967 | 34.4965 |
| 22       | 182.084 | 124.7   |

![image](/benchmark/fri/fri_benchmark_mac_m3.png)

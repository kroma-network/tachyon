# Poseidon2 Hash Benchmark

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
Note that Poseidon2 runs 10000x per test due to some time results being too small when running a single iteration.

## BN254

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p bn254_fr --vendor horizen --vendor plonky3 --check_results
```

### On Intel i9-13900K

| Trial Number | Tachyon  | Horizen      | Plonky3  |
| :----------- | -------- | ------------ | -------- |
| 0            | 0.069228 | **0.050903** | 0.085679 |
| 1            | 0.062046 | **0.050892** | 0.085772 |
| 2            | 0.06053  | **0.050848** | 0.08553  |
| 3            | 0.060648 | **0.050825** | 0.085643 |
| 4            | 0.060553 | **0.051126** | 0.08583  |
| 5            | 0.060592 | **0.05362**  | 0.085475 |
| 6            | 0.060576 | **0.050936** | 0.085731 |
| 7            | 0.06051  | **0.05081**  | 0.085613 |
| 8            | 0.060561 | **0.050889** | 0.086382 |
| 9            | 0.060558 | **0.050896** | 0.086557 |
| avg          | 0.06158  | **0.051174** | 0.085821 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_ubuntu_i9.png)

### On Mac M3 Pro

| Trial Number | Tachyon  | Horizen      | Plonky3  |
| :----------- | -------- | ------------ | -------- |
| 0            | 0.068967 | **0.058728** | 0.086994 |
| 1            | 0.068786 | **0.05839**  | 0.086825 |
| 2            | 0.068658 | **0.058245** | 0.086779 |
| 3            | 0.068673 | **0.058189** | 0.086693 |
| 4            | 0.068675 | **0.058303** | 0.08674  |
| 5            | 0.068693 | **0.058109** | 0.08681  |
| 6            | 0.068621 | **0.058405** | 0.086816 |
| 7            | 0.068747 | **0.058247** | 0.086871 |
| 8            | 0.068637 | **0.058383** | 0.086842 |
| 9            | 0.068665 | **0.058162** | 0.086846 |
| avg          | 0.068712 | **0.058316** | 0.086821 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_mac_m3.png)

## Baby Bear

Note: Horizen and Plonky3 compute values with a different internal matrix, requiring them to be compared with Tachyon separately.

### Horizen

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p baby_bear --vendor horizen --check_results
```

#### On Intel i9-13900K

| Trial Number | Tachyon      | Horizen  |
| :----------- | ------------ | -------- |
| 0            | **0.011424** | 0.034238 |
| 1            | **0.011975** | 0.034214 |
| 2            | **0.011505** | 0.034245 |
| 3            | **0.011304** | 0.03418  |
| 4            | **0.011313** | 0.034231 |
| 5            | **0.011354** | 0.034234 |
| 6            | **0.010743** | 0.034487 |
| 7            | **0.01071**  | 0.034259 |
| 8            | **0.010706** | 0.034229 |
| 9            | **0.010708** | 0.034246 |
| avg          | **0.011174** | 0.034256 |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon      | Horizen  |
| :----------- | ------------ | -------- |
| 0            | **0.010679** | 0.014511 |
| 1            | **0.010448** | 0.014653 |
| 2            | **0.010286** | 0.014961 |
| 3            | **0.01024**  | 0.014769 |
| 4            | **0.010233** | 0.014717 |
| 5            | **0.010267** | 0.014761 |
| 6            | **0.010226** | 0.01514  |
| 7            | **0.010303** | 0.01475  |
| 8            | **0.010253** | 0.014693 |
| 9            | **0.010326** | 0.014533 |
| avg          | **0.010326** | 0.014748 |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_mac_m3.png)

### Plonky3

```shell
GOMP_SPINCOUNT=0 bazel run --config maxopt --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p baby_bear --vendor plonky3 --check_results
```

#### On Intel i9-13900K

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.00999  | **0.005391** |
| 1            | 0.009882 | **0.005298** |
| 2            | 0.009848 | **0.00513**  |
| 3            | 0.009772 | **0.005157** |
| 4            | 0.00977  | **0.005072** |
| 5            | 0.009774 | **0.005032** |
| 6            | 0.009783 | **0.005062** |
| 7            | 0.009878 | **0.005077** |
| 8            | 0.009778 | **0.005014** |
| 9            | 0.009762 | **0.005016** |
| avg          | 0.009823 | **0.005124** |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.009116 | **0.007311** |
| 1            | 0.008967 | **0.007352** |
| 2            | 0.008805 | **0.007312** |
| 3            | 0.008748 | **0.007315** |
| 4            | 0.008742 | **0.007339** |
| 5            | 0.008741 | **0.007309** |
| 6            | 0.008774 | **0.00732**  |
| 7            | 0.00873  | **0.007696** |
| 8            | 0.008791 | **0.007342** |
| 9            | 0.008741 | **0.007353** |
| avg          | 0.008815 | **0.007364** |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_mac_m3.png)\*\*\*\*

# Poseidon2 Hash Benchmark

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

Note that Poseidon2 runs 10000x per test due to some time results being too small when running a single iteration.

## BN254

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p bn254_fr --vendor horizen --vendor plonky3 --check_results
```

### On Intel i9-13900K

| Trial Number | Tachyon  | Horizen      | Plonky3  |
| :----------- | -------- | ------------ | -------- |
| 0            | 0.064726 | **0.050047** | 0.082464 |
| 1            | 0.061723 | **0.049997** | 0.082598 |
| 2            | 0.060917 | **0.050063** | 0.08257  |
| 3            | 0.06086  | **0.049952** | 0.082493 |
| 4            | 0.060655 | **0.050173** | 0.082409 |
| 5            | 0.060768 | **0.050683** | 0.08268  |
| 6            | 0.060843 | **0.050278** | 0.082675 |
| 7            | 0.060696 | **0.050062** | 0.082579 |
| 8            | 0.060688 | **0.05004**  | 0.082592 |
| 9            | 0.060677 | **0.050128** | 0.083144 |
| avg          | 0.061255 | **0.050142** | 0.08262  |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_ubuntu_i9.png)

### On Mac M3 Pro

| Trial Number | Tachyon  | Horizen      | Plonky3  |
| :----------- | -------- | ------------ | -------- |
| 0            | 0.072979 | **0.055031** | 0.081624 |
| 1            | 0.072478 | **0.054731** | 0.081525 |
| 2            | 0.071973 | **0.055768** | 0.081633 |
| 3            | 0.071969 | **0.054835** | 0.081638 |
| 4            | 0.072009 | **0.054884** | 0.081545 |
| 5            | 0.071933 | **0.055**    | 0.081572 |
| 6            | 0.07201  | **0.054946** | 0.081521 |
| 7            | 0.072033 | **0.054883** | 0.081539 |
| 8            | 0.071967 | **0.054989** | 0.081626 |
| 9            | 0.071934 | **0.054942** | 0.081556 |
| avg          | 0.072128 | **0.055**    | 0.081577 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_mac_m3.png)

## Baby Bear

Note: Horizen and Plonky3 compute values with a different internal matrix, requiring them to be compared with Tachyon separately.

### Horizen

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p baby_bear --vendor horizen --check_results
```

#### On Intel i9-13900K

| Trial Number | Tachyon      | Horizen  |
| :----------- | ------------ | -------- |
| 0            | **0.011549** | 0.034751 |
| 1            | **0.011439** | 0.034627 |
| 2            | **0.011475** | 0.034581 |
| 3            | **0.011543** | 0.035442 |
| 4            | **0.011455** | 0.03632  |
| 5            | **0.011372** | 0.034545 |
| 6            | **0.011381** | 0.034538 |
| 7            | **0.011142** | 0.03459  |
| 8            | **0.010845** | 0.034522 |
| 9            | **0.010819** | 0.034589 |
| avg          | **0.011302** | 0.03485  |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon      | Horizen  |
| :----------- | ------------ | -------- |
| 0            | **0.010979** | 0.013892 |
| 1            | **0.010574** | 0.013858 |
| 2            | **0.010544** | 0.014098 |
| 3            | **0.010642** | 0.013843 |
| 4            | **0.010517** | 0.013842 |
| 5            | **0.010599** | 0.013938 |
| 6            | **0.010519** | 0.013913 |
| 7            | **0.010474** | 0.013889 |
| 8            | **0.010572** | 0.013892 |
| 9            | **0.010533** | 0.013979 |
| avg          | **0.010595** | 0.013914 |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_mac_m3.png)

### Plonky3

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p baby_bear --vendor plonky3 --check_results
```

#### On Intel i9-13900K

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.009848 | **0.006655** |
| 1            | 0.00992  | **0.006582** |
| 2            | 0.009955 | **0.006616** |
| 3            | 0.009811 | **0.006572** |
| 4            | 0.009851 | **0.006537** |
| 5            | 0.009776 | **0.006645** |
| 6            | 0.009822 | **0.006548** |
| 7            | 0.009738 | **0.006586** |
| 8            | 0.009757 | **0.006594** |
| 9            | 0.009717 | **0.006619** |
| avg          | 0.009819 | **0.006595** |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.009201 | **0.00732**  |
| 1            | 0.008927 | **0.007346** |
| 2            | 0.008922 | **0.007309** |
| 3            | 0.008735 | **0.007324** |
| 4            | 0.008765 | **0.0076**   |
| 5            | 0.008715 | **0.007335** |
| 6            | 0.008704 | **0.007439** |
| 7            | 0.008664 | **0.007369** |
| 8            | 0.008679 | **0.007347** |
| 9            | 0.008657 | **0.007353** |
| avg          | 0.008796 | **0.007374** |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_mac_m3.png)\*\*\*\*

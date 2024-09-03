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
| 0            | 0.06417  | **0.049875** | 0.082714 |
| 1            | 0.061176 | **0.049937** | 0.082634 |
| 2            | 0.060154 | **0.04997**  | 0.082722 |
| 3            | 0.060016 | **0.049925** | 0.082925 |
| 4            | 0.060068 | **0.049902** | 0.082724 |
| 5            | 0.060079 | **0.052291** | 0.082722 |
| 6            | 0.060046 | **0.051501** | 0.08271  |
| 7            | 0.06008  | **0.049871** | 0.082694 |
| 8            | 0.06005  | **0.049768** | 0.082677 |
| 9            | 0.060036 | **0.049995** | 0.083914 |
| avg          | 0.060587 | **0.050303** | 0.082843 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_ubuntu_i9.png)

### On Mac M3 Pro

| Trial Number | Tachyon  | Horizen      | Plonky3  |
| :----------- | -------- | ------------ | -------- |
| 0            | 0.073586 | **0.055299** | 0.082223 |
| 1            | 0.073077 | **0.055153** | 0.082353 |
| 2            | 0.072825 | **0.055762** | 0.082113 |
| 3            | 0.072752 | **0.054881** | 0.082102 |
| 4            | 0.072827 | **0.055061** | 0.082143 |
| 5            | 0.072827 | **0.054988** | 0.082222 |
| 6            | 0.072862 | **0.05494**  | 0.082222 |
| 7            | 0.072745 | **0.055217** | 0.082189 |
| 8            | 0.0729   | **0.054908** | 0.082151 |
| 9            | 0.072922 | **0.055135** | 0.082084 |
| avg          | 0.072932 | **0.055134** | 0.08218  |

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
| 0            | **0.012201** | 0.036443 |
| 1            | **0.012022** | 0.035775 |
| 2            | **0.011884** | 0.034381 |
| 3            | **0.011955** | 0.034372 |
| 4            | **0.011902** | 0.034503 |
| 5            | **0.011901** | 0.034408 |
| 6            | **0.011873** | 0.034462 |
| 7            | **0.011807** | 0.034393 |
| 8            | **0.011835** | 0.034304 |
| 9            | **0.011794** | 0.034446 |
| avg          | **0.011917** | 0.034748 |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon      | Horizen  |
| :----------- | ------------ | -------- |
| 0            | **0.011549** | 0.013957 |
| 1            | **0.01148**  | 0.014097 |
| 2            | **0.011512** | 0.01411  |
| 3            | **0.011383** | 0.013999 |
| 4            | **0.011461** | 0.014056 |
| 5            | **0.011371** | 0.014166 |
| 6            | **0.011386** | 0.014089 |
| 7            | **0.011587** | 0.014151 |
| 8            | **0.01141**  | 0.014135 |
| 9            | **0.011608** | 0.01411  |
| avg          | **0.011474** | 0.014087 |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_horizen_mac_m3.png)

### Plonky3

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p baby_bear --vendor plonky3 --check_results
```

#### On Intel i9-13900K

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.010727 | **0.006418** |
| 1            | 0.01039  | **0.006261** |
| 2            | 0.010394 | **0.006259** |
| 3            | 0.010416 | **0.006262** |
| 4            | 0.010356 | **0.006226** |
| 5            | 0.010302 | **0.006227** |
| 6            | 0.010302 | **0.006235** |
| 7            | 0.010301 | **0.006424** |
| 8            | 0.010116 | **0.006304** |
| 9            | 0.009905 | **0.006233** |
| avg          | 0.01032  | **0.006284** |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_ubuntu_i9.png)

#### On Mac M3 Pro

| Trial Number | Tachyon  | Plonky3      |
| :----------- | -------- | ------------ |
| 0            | 0.009962 | **0.007355** |
| 1            | 0.009939 | **0.007373** |
| 2            | 0.009851 | **0.007431** |
| 3            | 0.009943 | **0.007453** |
| 4            | 0.009853 | **0.007399** |
| 5            | 0.009826 | **0.007459** |
| 6            | 0.009825 | **0.007385** |
| 7            | 0.009884 | **0.007389** |
| 8            | 0.009819 | **0.007403** |
| 9            | 0.009937 | **0.007353** |
| avg          | 0.009883 | **0.0074**   |

![image](/benchmark/poseidon2/poseidon2_benchmark_baby_bear_plonky3_mac_m3.png)\*\*\*\*

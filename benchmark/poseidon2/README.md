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

Note that Poseidon2 runs 100x per test due to some time results being too small when running a single iteration.

## BN254

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p bn254_fr --vendor horizen --vendor plonky3 --check_results
```

### On Intel i9-13900K

| Trial Number | Tachyon   | Horizen       | Plonky3   |
| :----------- | --------- | ------------- | --------- |
| 0            | 0.000788  | **0.000534**  | 0.000876  |
| 1            | 0.000628  | **0.000585**  | 0.00087   |
| 2            | 0.000624  | **0.000517**  | 0.000865  |
| 3            | 0.000622  | **0.000513**  | 0.000866  |
| 4            | 0.000634  | **0.000603**  | 0.000861  |
| 5            | 0.000628  | **0.000512**  | 0.001002  |
| 6            | 0.000618  | **0.00051**   | 0.000853  |
| 7            | 0.000616  | **0.000553**  | 0.000852  |
| 8            | 0.0007    | **0.000693**  | 0.000873  |
| 9            | 0.000614  | **0.000525**  | 0.000937  |
| avg          | 0.0006472 | **0.0005545** | 0.0008855 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_ubuntu_i9.png)

### On Mac M3 Pro

| Trial Number | Tachyon   | Horizen       | Plonky3   |
| :----------- | --------- | ------------- | --------- |
| 0            | 0.001053  | **0.000816**  | 0.001186  |
| 1            | 0.001033  | **0.00076**   | 0.001177  |
| 2            | 0.001019  | **0.000726**  | 0.001157  |
| 3            | 0.001012  | **0.000712**  | 0.001172  |
| 4            | 0.001007  | **0.000691**  | 0.001152  |
| 5            | 0.001023  | **0.000684**  | 0.001131  |
| 6            | 0.001051  | **0.000682**  | 0.001123  |
| 7            | 0.001005  | **0.000678**  | 0.001116  |
| 8            | 0.000996  | **0.000687**  | 0.001118  |
| 9            | 0.001003  | **0.00068**   | 0.001127  |
| avg          | 0.0010202 | **0.0007116** | 0.0011459 |

![image](/benchmark/poseidon2/poseidon2_benchmark_bn254_mac_m3.png)

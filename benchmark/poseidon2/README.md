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

```shell
bazel run -c opt --//:has_openmp --//:has_rtti --//:has_matplotlib //benchmark/poseidon2:poseidon2_benchmark -- -p bn254_fr --vendor horizen --vendor plonky3 --check_results
```

#### On Intel i9-13900K

| Repetition | Tachyon | Horizen     | Plonky3 |
| :--------- | ------- | ----------- | ------- |
| 0          | 8e-06   | **7e-06**   | 1e-05   |
| 1          | 7e-06   | **5e-06**   | 8e-06   |
| 2          | 5e-06   | **4e-06**   | 8e-06   |
| 3          | 6e-06   | **4e-06**   | 8e-06   |
| 4          | 5e-06   | **3e-06**   | 7e-06   |
| 5          | 6e-06   | **3e-06**   | 7e-06   |
| 6          | 5e-06   | **3e-06**   | 7e-06   |
| 7          | 6e-06   | **3e-06**   | 7e-06   |
| 8          | 5e-06   | **4e-06**   | 7e-06   |
| 9          | 5e-06   | **3e-06**   | 7e-06   |
| avg        | 5.8e-06 | **3.9e-06** | 7.6e-06 |

![image](/benchmark/poseidon2/poseidon2_benchmark_ubuntu_i9.png)

#### On Mac M3 Pro

| Repetition | Tachyon | Horizen     | Plonky3  |
| :--------- | ------- | ----------- | -------- |
| 0          | 1.3e-05 | **1.2e-05** | 1.5e-05  |
| 1          | 1e-05   | **8e-06**   | 1.1e-05  |
| 2          | 9e-06   | **7e-06**   | 1e-05    |
| 3          | 9e-06   | **7e-06**   | 1e-05    |
| 4          | 9e-06   | **7e-06**   | 1e-05    |
| 5          | 9e-06   | **7e-06**   | 1e-05    |
| 6          | 9e-06   | **7e-06**   | 1e-05    |
| 7          | 9e-06   | **7e-06**   | 1e-05    |
| 8          | 9e-06   | **7e-06**   | 1e-05    |
| 9          | 9e-06   | **7e-06**   | 1e-05    |
| avg        | 9.5e-06 | **7.6e-06** | 1.06e-05 |

![image](/benchmark/poseidon2/poseidon2_benchmark_mac_m3.png)

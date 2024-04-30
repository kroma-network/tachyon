# Circom Benchmarking

```
Run on 13th Gen Intel(R) Core(TM) i9-13900K (32 X 5500 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
```

## How to run

Move to the `vendors/circom` directory.

```shell
cd vendors/circom
```

Run the following line if you are benchmarking for Circom for the first time.

```shell
CARGO_BAZEL_REPIN=1 bazel sync --only=crate_index
```

Run Circom benchmarking.

```shell
bazel run --@kroma_network_tachyon//:has_openmp -c opt //benchmark:circom_benchmark  -- -n 10
```

## Result

```
[0]: 0.044026 s
[1]: 0.039156 s
[2]: 0.038875 s
[3]: 0.038884 s
[4]: 0.038806 s
[5]: 0.038947 s
[6]: 0.038836 s
[7]: 0.038895 s
[8]: 0.038878 s
[9]: 0.038811 s
tachyon(avg): 0.039411 s
[0]: 0.530905 s
[1]: 0.75682 s
[2]: 0.163237 s
[3]: 0.77573 s
[4]: 0.530742 s
[5]: 0.512661 s
[6]: 0.22959 s
[7]: 0.542708 s
[8]: 0.72599 s
[9]: 0.490333 s
rapidsnark(avg): 0.525871 s
```

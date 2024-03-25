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
[0]: 0.051697 s
[1]: 0.046485 s
[2]: 0.046371 s
[3]: 0.046311 s
[4]: 0.059727 s
[5]: 0.046551 s
[6]: 0.046426 s
[7]: 0.04638 s
[8]: 0.046409 s
[9]: 0.046471 s
tachyon(avg): 0.048282 s
[0]: 0.32005 s
[1]: 0.50119 s
[2]: 0.252181 s
[3]: 0.6587 s
[4]: 0.747932 s
[5]: 0.743909 s
[6]: 0.663381 s
[7]: 0.432438 s
[8]: 0.420855 s
[9]: 0.698306 s
rapidsnark(avg): 0.543894 s
```

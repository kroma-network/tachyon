# Benchmark

Tachyon offers a suite of benchmarking tools and scripts designed to evaluate the performance of cryptographic and mathematical operations. These benchmarks span both functionalities inherent to Tachyon and those provided by other libraries. Our aim is to present a comparative landscape, assisting both users and developers in grasping the nuances of performance trade-offs.

## Benchmarks Included

1. EC Benchmark
   - This benchmark gauges the efficiency of elliptic curve operations on both CPU and GPU.
   - Delve into detailed insights about point multiplications, additions, and other fundamental elliptic curve operations.
2. MSM Benchmark
   - Multi-scalar multiplication (MSM) plays a pivotal role in cryptographic protocols. This benchmark allows you to gauge its performance on both CPU and GPU platforms.
   - Additionally, you can compare Tachyon's MSM performance against a range of other external libraries.

### Running the Benchmark

```shell
bazel run -c opt //benchmark/path/to/target:target_name -- -n <test_set_size>
```

- The `-n` flag designates the test set size. A test set size of n translates to n for the EC benchmark and 2^n for the MSM benchmark. Benchmarks can be executed for individual or multiple test set sizes.

- Use the --vendor flag to designate the benchmark target (only for MSM). Available vendors include: arkworks, bellman and halo2.

For example, to benchmark Arkworks MSM for test set sizes of 2^10, 2^11, and 2^12:

```shell
bazel run -c opt //benchmark/msm:msm_benchmark -- -n 10 -n 11 -n 12 --vendor arkworks
```

### Additional Options

By default, targets with the `cuda` or `rust` flags are excluded from the build target. If you wish to benchmark these targets, you'll need to enable them in your `.bazelrc.user`:

```
build --build_tag_filters -cuda //enables rust targets
build --build_tag_filters -rust //enables cuda targets
build --build_tag_filters="" //enables all targets
```

To harness the plot chart feature, first ensure that matplotlib is installed (refer to the [installation guide](/docs/how_to_use/how_to_build.md#matplotliboptional)). Then, append the `--//:has_rtti` and `--//:has_matplotlib` flags to your command:

```shell
bazel run -c opt --//:has_rtti --//:has_matplotlib //benchmark/path/to/target:target_name -- -n <test_set_size>
```

For executing GPU benchmarks, make sure to configure [GPU config](https://github.com/kroma-network/tachyon#hardware-acceleration) for your environment. For instance, in CUDA:

```shell
bazel run -c opt --config cuda //benchmark/path/to/target:target_name -- -n <test_set_size>
```

While ensuring computation accuracy is paramount, verifying results should ideally be reserved for testing phases due to potential time constraints. In some cases, checking for the equality of results can be time-consuming. Therefore, by default, this verification is disabled.

However, if you wish to validate the consistency of MSM computation results across different targets, you can enable this verification using the --check_results option:

```shell
bazel run -c opt //benchmark/msm:msm_benchmark -- -n <test_set_size> --vendor <benchmark_target> --check_results
```

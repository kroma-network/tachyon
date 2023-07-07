# Tachyon

This is a ZKP accelerator using CUDA.

## Prerequisites

### Bazel

Please follow the instructions [here](https://bazel.build/install).

### Ubuntu

```shell
> sudo apt install libgmp-dev
```

### Macos

```shell
> brew install gmp
```

## Getting started

### Build

#### Build on Linux

```shell
> bazel build --config linux //...
```

#### Build on Macos

```shell
> bazel build --config macos //...
```

### Test

#### Test on Linux

```shell
> bazel test --config linux //...
```

#### Test on Macos

```shell
> bazel test --config macos //...
```

## Configuration

- `--config gmp_backend`: Enable [gmp](https://gmplib.org/) prime field backend.

   ```shell
   > bazel build --config gmp_backend //...
   ```

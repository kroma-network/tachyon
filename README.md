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

## Build

```shell
> bazel build //...
```

## Test

```shell
> bazel test //...
```

## Configuration

- `--config gmp_backend`: Enable [gmp](https://gmplib.org/) prime field backend.

   ```shell
   > bazel build --config gmp_backend //...
   ```

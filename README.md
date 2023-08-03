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

### GMP backend prime field

- `--config gmp_backend`: Enable [gmp](https://gmplib.org/) prime field backend.

   ```shell
   > bazel build --config ${os} --config gmp_backend //...
   ```

- `--config goldilocks_backend`: Enable [goldilocks](https://github.com/0xPolygonHermez/goldilocks) prime field backend.

   ```shell
   > bazel build --config ${os} --config goldilocks_backend //...
   ```

### Hardware acceleration

#### CUDA backend

- `--config cuda`: Enable [cuda] backend.

   ```shell
   > bazel build --config ${os} --config cuda //...
   ```

- `--config rocm`: Enable [rocm] backend.

   ```shell
   > bazel build --config ${os} --config rocm //...
   ```

[cuda]: https://developer.nvidia.com/cuda-toolkit
[rocm]: https://www.amd.com/en/graphics/servers-solutions-rocm

### Matplotlib

### Pyenv

If you are using pyenv, don't forget to add a option `--enable-shared`.

```shell
> CONFIGURE_OPTS=--enable-shared pyenv install <version>
```

### Python dependencies

```shell
> pip install matplotlib
```

### Frequently Asked Questions

#### Debugging on macOS

Please add this line to your `.bazelrc.user`.

```
build --spawn_strategy=local
```

#### Build on Ubuntu 20.04

Please update g++ version and try build again! The default `g++-9` is not working.

```shell
> sudo apt install g++-10
> export CC=/usr/bin/g++-10
```

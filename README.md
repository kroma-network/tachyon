<div align="center">
  <br /><br /><br />
  <img src="tachyon_logo_horizontal.png" style="width: 50%;">
  <br /><br /><br />
</div>

Tachyon is a ZKP accelerator using CUDA.

## Prerequisites

### Bazel

Please follow the instructions [here](https://bazel.build/install).

### Ubuntu

```shell
> sudo apt install libgmp-dev libomp-dev
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

#### Build on Macos arm64

```shell
> bazel build --config macos_arm64 //...
```

#### Build on Macos x64

```shell
> bazel build --config macos_x86_64 //...
```

### Test

#### Test on Linux

```shell
> bazel test --config linux //...
```

#### Test on Macos arm64

```shell
> bazel test --config macos_arm64 //...
```

#### Test on Macos x64

```shell
> bazel test --config macos_x86_64 //...
```

## Configuration

### GMP backend prime field

- `--//:gmp_backend`: Enable [gmp](https://gmplib.org/) prime field backend.

  ```shell
  > bazel build --//:${os} --//:gmp_backend //...
  ```

### Polygon zkEVM backend prime field

_NOTE:_: Only x86_64 is supported.

- `--//:polygon_zkevm_backend`: Enable [goldilocks](https://github.com/0xPolygonHermez/goldilocks) and [zkevm-prover](https://github.com/0xPolygonHermez/zkevm-prover) prime field backend.

  ```shell
  > bazel build --config ${os} --config avx512_${os} --//:polygon_zkevm_backend //...
  ```

### Hardware acceleration

#### CUDA backend

- `--config cuda`: Enable [cuda] backend.

  ```shell
  > bazel build --config ${os} --config cuda //...
  ```

#### ROCm backend

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
> export CC=/usr/bin/gcc-10
> export CXX=/usr/bin/g++-10
> export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-10
```

#### Build CUDA with rust toolchain

```shell
Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging
error: linking with `external/local_config_cuda/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc` failed: exit status: 127
...
  = note: /usr/bin/env: 'python': No such file or directory
```

Please make your `python` point to python interpreter to be run.

```shell
> sudo apt install python-is-python3
```

Plus, please include these lines to your `.bazelc.user`.

```
build --action_env=PATH=/usr/bin:/usr/local/bin
build --host_action_env=PATH=/usr/bin:/usr/local/bin
```

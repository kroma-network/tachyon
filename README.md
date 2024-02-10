<div align="center">
  <br /><br /><br />
  <img src="tachyon_logo_horizontal.png" style="width: 50%;">
  <br /><br /><br />
</div>

## Overview

**Tachyon** is a Modular ZK Backend, powered by GPU.

## Design Goals

1. General Purpose: A versatile ZK library empowers developers to implement any proving scheme with minimal effort, often enhancing developer productivity. To create a general-purpose backend, aligning the code structure as closely as possible with the algebraic structure is paramount.
2. Easy to Use: Achieving widespread adoption is essential for the success of any product. Consequently, one of the key focal points of the Tachyon project is to include offering packages for various programming languages and runtimes.
3. Blazing Fast: Tachyon's foremost requirement is speed, and not just any speed, but blazing speed! This entails Tachyon delivering exceptional performance on both CPU and GPU platforms.
4. GPU Interoperability: Tachyon's code is designed to be compatible with both CPU and GPU in the majority of scenarios.

## List of Features

Symbol Definitions:

- :heavy_check_mark: Currently supported.
- 🏗️ Partially implemented or is under active construction.
- :x: Not currently supported.

### Finite Fields

|            | CPU                | GPU |
| ---------- | ------------------ | --- |
| Goldilocks | :heavy_check_mark: | :x: |

### Elliptic Curves

|           | CPU                | GPU                |
| --------- | ------------------ | ------------------ |
| bn254     | :heavy_check_mark: | :heavy_check_mark: |
| bls12-381 | :heavy_check_mark: | :heavy_check_mark: |
| secp256k1 | :heavy_check_mark: | :heavy_check_mark: |

### Commitment Schemes

|          | CPU                | GPU |
| -------- | ------------------ | --- |
| GWC      | :heavy_check_mark: | 🏗️  |
| SHPlonk  | :heavy_check_mark: | 🏗️  |
| FRI      | :heavy_check_mark: | :x: |
| Pedersen | :heavy_check_mark: | :x: |

### Hashes

|          | CPU                | GPU |
| -------- | ------------------ | --- |
| Poseidon | :heavy_check_mark: | :x: |

### Lookups

|       | CPU                | GPU |
| ----- | ------------------ | --- |
| Halo2 | :heavy_check_mark: | :x: |

### SNARKs

|       | CPU                | GPU |
| ----- | ------------------ | --- |
| Halo2 | :heavy_check_mark: | :x: |

### Frontends

|        | CPU                | GPU |
| ------ | ------------------ | --- |
| Circom | 🏗️                 | :x: |
| Halo2  | :heavy_check_mark: | :x: |

## Roadmap

- [x] 2024Q1 - Enable producing the [zkEVM](https://github.com/kroma-network/zkevm-circuits) proof.
- [ ] 2024Q2 - Replace Halo2 with Tachyon in [Kroma](https://kroma.network/) mainnet.
- [ ] 2024Q3 - Implement Halo2 GPU.
- [ ] 2024Q4 - Implement Halo2 Folding Scheme.

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

**_NOTE:_: MacOS v14.0.0 or later is recommended.**

In certain versions of MacOS (prior to v13.5.1), a bug related to incorrect Bigint divide operations has been detected in the field generator when using the optimized build (`-c opt`).

The [issue](https://github.com/kroma-network/tachyon/issues/98) will be fixed as soon as possible.

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

If you are using pyenv, don't forget to add an option `--enable-shared`.

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

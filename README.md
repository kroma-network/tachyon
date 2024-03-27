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

|             | CPU                | GPU |
| ----------- | ------------------ | --- |
| Goldilocks  | :heavy_check_mark: | :x: |
| Mersenne-31 | :heavy_check_mark: | :x: |

### Elliptic Curves

|           | CPU                | GPU                |
| --------- | ------------------ | ------------------ |
| bn254     | :heavy_check_mark: | :heavy_check_mark: |
| bls12-381 | :heavy_check_mark: | :heavy_check_mark: |
| secp256k1 | :heavy_check_mark: | :heavy_check_mark: |
| pallas    | :heavy_check_mark: | :heavy_check_mark: |
| vesta     | :heavy_check_mark: | :heavy_check_mark: |

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

|         | CPU                | GPU |
| ------- | ------------------ | --- |
| Groth16 | :heavy_check_mark: | :x: |
| Halo2   | :heavy_check_mark: | :x: |

### Frontends

|                 | CPU                | GPU |
| --------------- | ------------------ | --- |
| Circom(groth16) | :heavy_check_mark: | :x: |
| Halo2           | :heavy_check_mark: | :x: |

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
sudo apt install libgmp-dev libomp-dev
```

### Macos

```shell
brew install gmp
```

## Getting started

### Build

```shell
bazel build --config {os} //...
```

### Test

```shell
bazel test --config {os} //...
```

Check [How To Build](/docs/how_to_use/how_to_build.md) for more information.

# How to Build

You can build Tachyon with this guide. The instructions below will help you set up your environment, install necessary dependencies, and compile your projects efficiently. Follow along to start building with confidence.

## Prerequisites

### Bazel

Follow the installation instructions for Bazel [here](https://bazel.build/install).

### Requirements for each OS

#### Ubuntu

```shell
sudo apt install libgmp-dev libomp-dev zip
```

#### MacOS

```shell
brew install gmp
```

### Python dependencies

#### Python

- Install [Python](https://www.python.org/downloads/). Python **v3.10.12** is recommended.

- If you are using pyenv, don't forget to add an option `--enable-shared`.

  ```shell
  CONFIGURE_OPTS=--enable-shared pyenv install <version>
  ```

#### Matplotlib(optional)

If you want to print a benchmark plot, you need to install matplotlib. See [here](/benchmark/README.md) for more details.

```shell
pip install matplotlib
```

**_NOTE_:**

- To build `mac_logging` in MacOS, you need to install objc, which can be done by installing XCode.

- MacOS v14.0.0 or later is recommended. In certain versions of MacOS (prior to v13.5.1), a bug related to incorrect `BigInt` divide operations has been detected in the field generator when using the optimized build (`-c opt`). This [issue](https://github.com/kroma-network/tachyon/issues/98) will be fixed as soon as possible.

## Build from source

### Build

#### Build on Linux

```shell
bazel build --config linux //...
```

#### Build on Macos arm64

```shell
bazel build --config macos_arm64 //...
```

#### Build on Macos x64

```shell
bazel build --config macos_x86_64 //...
```

### Test

#### Test on Linux

```shell
bazel test --config linux //...
```

#### Test on Macos arm64

```shell
bazel test --config macos_arm64 //...
```

#### Test on Macos x64

```shell
bazel test --config macos_x86_64 //...
```

## Configurations

### OS options

- `linux`
- `macos_x86_64`
- `macos_arm64`

```shell
bazel build --config macos_arm64 //...
```

### Build options

- `opt`: Default optimized build option
- `dbg`: Build with debug info
- `fastbuild`: Fast build option

```shell
bazel build --config macos_arm64 --config dbg //...
```

### Hardware acceleration

#### CUDA backend

- `--config cuda`: Enable [cuda] backend.

  ```shell
  bazel build --config ${os} --config cuda //...
  ```

#### ROCm backend

- `--config rocm`: Enable [rocm] backend.

  ```shell
  bazel build --config ${os} --config rocm //...
  ```

_NOTE_: The `rocm` option is not recommended for current use because it is not being tested yet.

[cuda]: https://developer.nvidia.com/cuda-toolkit
[rocm]: https://www.amd.com/en/graphics/servers-solutions-rocm

### .bazelrc.user

Build options can be preset in `.bazelrc.user` for your convenience, eliminating the need to specify them on the command line.

For example:

```
# .bazelrc.user

build --config linux
build --config dbg
```

```shell
bazel build //...
# With the preset options in .bazelrc.user, this is the same as:
# bazel build --config linux --config dbg //...
```

## Building Tachyon from a Bazel repository

Tachyon can be built in your own Bazel project with the following two simple steps.

First, obtain the Tachyon code from a specific commit hash and get a SHA256 value from the fetched code through these commands:

```shell
wget https://github.com/kroma-network/tachyon/archive/79ec2875bdc6fce62e490c845b44477b15df45ec.tar.gz

shasum -a 256 79ec2875bdc6fce62e490c845b44477b15df45ec.tar.gz
```

Second, input the shasum output into your `WORKSPACE` file as the `sha256` argument like shown below:

```bzl
# WORKSPACE

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "kroma_network_tachyon",
    sha256 = "a37fba8082a2c27a4a2f09255cf6c2693836dccbb8a8cea7e327d860115ffd30",
    strip_prefix = "tachyon-79ec2875bdc6fce62e490c845b44477b15df45ec",
    urls = ["https://github.com/kroma-network/tachyon/archive/79ec2875bdc6fce62e490c845b44477b15df45ec.tar.gz"],
)
```

## Debian packaging

There are two ways to install the Tachyon package. While it is recommended to install the package from the prebuilt binaries, building the package from source is also viable if needed.

### Install package from pre-built binaries

```shell
curl -LO https://github.com/kroma-network/tachyon/releases/download/v0.2.0/libtachyon_0.2.0_amd64.deb
curl -LO https://github.com/kroma-network/tachyon/releases/download/v0.2.0/libtachyon-dev_0.2.0_amd64.deb

sudo dpkg -i libtachyon_0.2.0_amd64.deb
sudo dpkg -i libtachyon-dev_0.2.0_amd64.deb
```

### Build package from source

Build a Debian package with the supported scheme (only halo2 for now) and the options you want.
To build the Halo2 Debian package, `halo2` and `has_openmp` options are recommended. Run the following commands:

```shell
bazel build -c opt --config ${os} --config halo2 --//:has_openmp  --//:c_shared_object //scripts/packages/debian/runtime:debian
bazel build -c opt --config ${os} --config halo2 --//:has_openmp  --//:c_shared_object //scripts/packages/debian/dev:debian

sudo dpkg -i bazel-bin/scripts/packages/debian/runtime/libtachyon_0.2.0_amd64.deb
sudo dpkg -i bazel-bin/scripts/packages/debian/dev/libtachyon-dev_0.2.0_amd64.deb
```

## Other Info

### Debugging on macOS

Please add this line to your `.bazelrc.user`.

```
build --spawn_strategy=local
```

### Control build && test

As you can see in [.bazelrc](/.bazelrc), we don't enable all of the build and test targets. To enable additional targets, you may add customized `--build_tag_filters` and `--test_tag_filters` to your `.bazelrc.user` file.

For example, add the following to `.bazelrc.user` to build rust on linux:

```
build --build_tag_filters -objc,-cuda
```

### Build SP1 Rust code

Since [scale-info](https://crates.io/crates/scale-info) needs a `CARGO` environment variable, add this to your `.bazelrc.user`.

```
build --action_env=CARGO=<path to cargo>
```

### Py Binding

`ModuleNotFoundError` may occur in certain python versions (v3.11.6). Python v3.10.12 is recommended.

### Build on Ubuntu 20.04

If you are using Ubuntu 20.04, update your g++ version. The default `g++-9` does not work.

```shell
sudo apt install g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-10
```

### Building with Clang on Ubuntu 22.04

To utilize features like [always_inline](https://releases.llvm.org/15.0.0/tools/clang/docs/AttributeReference.html#always-inline-force-inline),
you'll need clang++ at version 15 or higher. By default, Ubuntu 22.04 installs `clang-14` via `apt install clang`,
so you'll need to manually install a newer version of clang and update `openmp` accordingly.

First, install `clang-15` and the appropriate OpenMP libraries:

```shell
sudo apt install clang-15 libomp5-15 libomp-15-dev
```

Next, update `CC` and `CXX` to point to the newly installed `clang-15` and `clang++-15`:

```shell
export CC=/usr/bin/clang-15
export CXX=/usr/bin/clang++-15
```

Make sure `CC` and `CXX` are properly updated to `clang-15` and `clang++-15`.
These commands ensure that your build environment uses clang version 15 for compilation.

### Build CUDA with rust toolchain

You may run into the following problem:

```shell
Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging
error: linking with `external/local_config_cuda/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc` failed: exit status: 127
...
  = note: /usr/bin/env: 'python': No such file or directory
```

If so, make your `python` point to the python interpreter.

```shell
sudo apt install python-is-python3
```

Additionally, please include these lines in your `.bazelc.user`.

```
build:cuda --action_env=PATH=/usr/bin:/usr/local/bin
build:cuda --host_action_env=PATH=/usr/bin:/usr/local/bin
```

### Generate C API documents using Doxygen

Doxygen generates C API documents (`tachyon_api_docs.zip`) in the binary directory (`bazel-bin/docs/doxygen/`). Currently this feature is available only on Linux.

```shell
bazel build --config linux //docs/doxygen:generate_docs &&
unzip -o bazel-bin/docs/doxygen/tachyon_api_docs.zip &&
google-chrome bazel-bin/docs/doxygen/html/index.html
# generate HTML files and open on Chrome browser.
```

### Build without assembly-optimized prime field

You may encounter an illegal instruction error when running unit tests. Although the exact cause is not yet known, thanks to @zkbitcoin, we discovered that this issue is related to [ffiasm](https://github.com/iden3/ffiasm). Temporarily disabling assembly-optimized prime field can resolve this problem.

```shell
bazel build --config ${os} --//:has_asm_prime_field=false //...
```

## Performance Tuning

### Use Intel OpenMP Runtime Library(libiomp)

By default, Tachyon uses GNU OpenMP (GNU `libgomp`) for parallel computation. On Intel platforms, Intel OpenMP Runtime Library (`libiomp`) provides OpenMP API specification support. It sometimes brings more performance benefits compared to `libgomp`.

You can install `libomp` by following the [instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html).

To link `libomp`, you need to add an additional flag `--//:has_intel_openmp`.

```shell
bazel build --config ${os} --//:has_openmp --//:has_intel_openmp //...
```

See also [PyTorch Recipes/Performance Tuning Guide/Intel OpenMP Library](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#intel-openmp-runtime-library-libiomp).

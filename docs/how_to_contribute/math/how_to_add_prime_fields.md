# How to add prime fields

Follow this guide to add a new prime field for Tachyon.

_Note_: We have our own development conventions. Please read the [conventions doc](/docs/how_to_contribute/conventions.md) before contributing.

## 1. Add a `BUILD.bazel` file

Begin by creating a directory named `<new_prime_field>_prime` in `/tachyon/math/finite_field/`. Add a `BUILD.bazel` file into this directory. Note that once parameters are added to `BUILD.bazel`, Bazel will automatically generate the prime field code based on these parameters when it builds the target.

Choose among `generate_prime_fields()`, `generate_fft_prime_fields()`, or `generate_large_fft_prime_fields()` for code generation, depending on the prime field type. For more information, refer to [`prime_field_generator`](/tachyon/math/finite_fields/generator/prime_field_generator/build_defs.bzl).

For instance, to implement a FFT prime field, create a directory (`/tachyon/math/finite_fields/<new_prime_field>`) and add a `BUILD.bazel` file as shown below:

```bazel
# /tachyon/math/finite_fields/<new_prime_field>/BUILD.bazel

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")
load(
    "//tachyon/math/finite_fields/generator/prime_field_generator:build_defs.bzl",
    "SUBGROUP_GENERATOR",
    "generate_fft_prime_fields", # NOTE: Choose generator type
)

package(default_visibility = ["//visibility:public"])

string_flag(
    name = SUBGROUP_GENERATOR,
    build_setting_default = "{subgroup_generator}", # input Subgroup generator value
)

generate_fft_prime_fields( # NOTE: Choose generator type
    name = "new_prime_field",
    class_name = "NewPrimeField",
    modulus = "{modulus_value}", # input modulus value
    namespace = "tachyon::math",
    subgroup_generator = ":" + SUBGROUP_GENERATOR,
)

tachyon_cc_library(
    name = "prime_field_<new_prime_field>",
    hdrs = ["prime_field_<new_prime_field>.h"],
    defines = ["TACHYON_POLYGON_ZKEVM_BACKEND"],
    deps = [
        "//tachyon/math/base/gmp:gmp_util",
        "//tachyon/math/finite_fields:prime_field_base",
    ],
)
```

Use the following files for reference:

- [Goldilocks `BUILD.bazel`](/tachyon/math/finite_fields/goldilocks_prime/BUILD.bazel)
- [Mersenne-31 `BUILD.bazel`](/tachyon/math/finite_fields/mersenne_31_prime/BUILD.bazel)

## 2. Add the header file

Next, implement the logic for the new prime field in a header file named `prime_field_<prime_field_name>.h`. The code will then be generated in `bazel-bin/tachyon/math/finite_fields/<new_prime_field_dir>/<prime_field_name>.h` when Bazel builds the correlating target.

Use the following files for reference:

- [`prime_field_goldilocks.h`](/tachyon/math/finite_fields/goldilocks_prime/prime_field_goldilocks.h)
- [`prime_field_mersenne_31.h`](/tachyon/math/finite_fields/mersenne_31_prime/prime_field_mersenne_31.h)

## 3. Add to `prime_field_generator_unittest.cc`

Finally, ensure the prime field works well by incorporating it into `prime_field_generator_unittest.cc` at the two points shown below:

```cpp
...
#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"
#include "tachyon/math/finite_fields/mersenne_31_prime/mersenne_31.h"
// ADD NEW PRIME FIELD HEADER FILE HERE
// #include "tachyon/math/finite_fields/..."
...
// ADD NEW PRIME FIELD NAME HERE
using PrimeFieldTypes =
    testing::Types<bls12_381::Fq, bls12_381::Fr, bn254::Fq, bn254::Fr,
                   secp256k1::Fr, secp256k1::Fq, Goldilocks, Mersenne31 /*, NEW PRIME FIELD*/>;
...
```

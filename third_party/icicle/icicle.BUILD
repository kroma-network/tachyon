load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@kroma_network_tachyon//bazel:tachyon.bzl", "if_gpu_is_configured")
load("@kroma_network_tachyon//bazel:tachyon_cc.bzl", "tachyon_cuda_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(default_visibility = ["//visibility:public"])

VALUES = [
    "bn254",
    "bls12_381",
    "bls12_377",
    "bw6_761",
    "grumpkin",
    "baby_bear",
    "stark_252",
]

string_flag(
    name = "field_id",
    build_setting_default = "bn254",
    values = VALUES,
)

[
    config_setting(
        name = "{}_field_id".format(value),
        flag_values = {":field_id": value},
    )
    for value in VALUES
]

tachyon_cuda_library(
    name = "hdrs",
    hdrs = glob(["icicle/include/**/*.h"]),
    defines = select({
        ":bn254_field_id": [
            "FIELD_ID=BN254",
            "CURVE_ID=BN254",
            "CURVE=bn254",
        ],
        ":bls12_381_field_id": [
            "FIELD_ID=BLS12_381",
            "CURVE_ID=BLS12_381",
            "CURVE=bls12_381",
        ],
        ":bls12_377_field_id": [
            "FIELD_ID=BLS12_377",
            "CURVE_ID=BLS12_377",
            "CURVE=bls12_377",
        ],
        ":bw6_761_field_id": [
            "FIELD_ID=BW6_761",
            "CURVE_ID=BW6_761",
            "CURVE=bw6_761",
        ],
        ":grumpkin_field_id": [
            "FIELD_ID=GRUMPKIN",
            "CURVE_ID=GRUMPKIN",
            "CURVE=grumpkin",
        ],
        ":baby_bear_field_id": ["FIELD_ID=BABY_BEAR"],
        ":stark_252_field_id": ["FIELD_ID=STARK_252"],
        "//conditions:default": [],
    }),
    include_prefix = "third_party/icicle/include",
    includes = ["icicle/include"],
    strip_include_prefix = "icicle/include",
    deps = if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

tachyon_cuda_library(
    name = "curves",
    srcs = if_gpu_is_configured([
        "icicle/src/curves/extern.cu.cc",
        "icicle/src/curves/extern_g2.cu.cc",
    ]),
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/curves"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "fields",
    srcs = if_gpu_is_configured([
        "icicle/src/fields/extern.cu.cc",
        "icicle/src/fields/extern_extension.cu.cc",
    ]),
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/fields"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "hash",
    srcs = if_gpu_is_configured([
        "icicle/src/hash/keccak/keccak.cu.cc",
    ]),
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/hash"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "msm",
    srcs = if_gpu_is_configured([
        "icicle/src/msm/extern.cu.cc",
        "icicle/src/msm/extern_g2.cu.cc",
    ]),
    hdrs = ["icicle/src/msm/msm.cu.cc"],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/msm"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "ntt",
    srcs = if_gpu_is_configured([
        "icicle/src/ntt/extern_ecntt.cu.cc",
        "icicle/src/ntt/extern.cu.cc",
        "icicle/src/ntt/extern_extension.cu.cc",
        "icicle/src/ntt/kernel_ntt.cu.cc",
    ]),
    hdrs = [
        "icicle/src/ntt/ntt.cu.cc",
        "icicle/src/ntt/thread_ntt.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/ntt"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "polynomials",
    srcs = if_gpu_is_configured([
        "icicle/src/polynomials/polynomials.cu.cc",
        "icicle/src/polynomials/polynomials_c_api.cu.cc",
        "icicle/src/polynomials/cuda_backend/polynomial_cuda_backend.cu.cc",
    ]),
    hdrs = ["icicle/src/polynomials/cuda_backend/kernels.cu.h"],
    include_prefix = "third_party/icicle/src",
    includes = ["includes/src/polynomials"],
    strip_include_prefix = "icicle/src",
    deps = [
        ":hdrs",
        ":vec_ops",
    ],
)

tachyon_cuda_library(
    name = "poseidon",
    srcs = if_gpu_is_configured([
        "icicle/src/poseidon/tree/merkle.cu.cc",
        "icicle/src/poseidon/extern.cu.cc",
    ]),
    hdrs = [
        "icicle/src/poseidon/constants.cu.cc",
        "icicle/src/poseidon/kernels.cu.cc",
        "icicle/src/poseidon/poseidon.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["includes/src/poseidon"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "poseidon2",
    srcs = if_gpu_is_configured([
        "icicle/src/poseidon2/extern.cu.cc",
    ]),
    hdrs = [
        "icicle/src/poseidon2/constants.cu.cc",
        "icicle/src/poseidon2/kernels.cu.cc",
        "icicle/src/poseidon2/poseidon.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["includes/src/poseidon2"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

tachyon_cuda_library(
    name = "vec_ops",
    srcs = if_gpu_is_configured([
        "icicle/src/vec_ops/extern_extension.cu.cc",
        "icicle/src/vec_ops/extern.cu.cc",
    ]),
    hdrs = ["icicle/src/vec_ops/vec_ops.cu.cc"],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/vec_ops"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

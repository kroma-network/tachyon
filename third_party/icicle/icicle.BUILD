load(
    "@icicle//:build_defs.bzl",
    "CURVES",
    "FIELDS",
    "FIELDS_WITH_MERKLE_TREE",
    "FIELDS_WITH_NTT",
    "FIELDS_WITH_POSEIDON",
    "FIELDS_WITH_POSEIDON2",
    "icicle_defines",
)
load("@kroma_network_tachyon//bazel:tachyon.bzl", "if_gpu_is_configured")
load("@kroma_network_tachyon//bazel:tachyon_cc.bzl", "tachyon_cuda_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(default_visibility = ["//visibility:public"])

tachyon_cuda_library(
    name = "hdrs",
    hdrs = glob(["icicle/include/**/*.h"]),
    include_prefix = "third_party/icicle/include",
    includes = ["icicle/include"],
    strip_include_prefix = "icicle/include",
    deps = if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
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

[tachyon_cuda_library(
    name = "merkle_tree_{}".format(field),
    hdrs = [
        "icicle/src/merkle-tree/merkle.cu.cc",
        "icicle/src/merkle-tree/mmcs.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/merkle-tree"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
) for field in FIELDS_WITH_MERKLE_TREE]

[tachyon_cuda_library(
    name = "msm_{}".format(field),
    hdrs = ["icicle/src/msm/msm.cu.cc"],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/msm"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
) for field in CURVES]

[tachyon_cuda_library(
    name = "ntt_{}".format(field),
    srcs = if_gpu_is_configured([
        "icicle/src/ntt/kernel_ntt.cu.cc",
    ]),
    hdrs = [
        "icicle/src/ntt/ntt.cu.cc",
        "icicle/src/ntt/thread_ntt.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/ntt"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
) for field in FIELDS_WITH_NTT]

[tachyon_cuda_library(
    name = "polynomials_{}".format(field),
    srcs = if_gpu_is_configured([
        "icicle/src/polynomials/polynomials.cu.cc",
        "icicle/src/polynomials/polynomials_c_api.cu.cc",
        "icicle/src/polynomials/cuda_backend/polynomial_cuda_backend.cu.cc",
    ]),
    hdrs = ["icicle/src/polynomials/cuda_backend/kernels.cu.h"],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/polynomials"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [
        ":hdrs",
        ":vec_ops",
    ],
) for field in FIELDS]

[tachyon_cuda_library(
    name = "poseidon_{}".format(field),
    srcs = if_gpu_is_configured([
        "icicle/src/poseidon/tree/merkle.cu.cc",
    ]),
    hdrs = [
        "icicle/src/poseidon/constants.cu.cc",
        "icicle/src/poseidon/kernels.cu.cc",
        "icicle/src/poseidon/poseidon.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/poseidon"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
) for field in FIELDS_WITH_POSEIDON]

[tachyon_cuda_library(
    name = "poseidon2_{}".format(field),
    hdrs = [
        "icicle/src/poseidon2/constants.cu.cc",
        "icicle/src/poseidon2/kernels.cu.cc",
        "icicle/src/poseidon2/poseidon.cu.cc",
    ],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/poseidon2"],
    local_defines = icicle_defines(field),
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
) for field in FIELDS_WITH_POSEIDON2]

tachyon_cuda_library(
    name = "vec_ops",
    hdrs = ["icicle/src/vec_ops/vec_ops.cu.cc"],
    include_prefix = "third_party/icicle/src",
    includes = ["icicle/src/vec_ops"],
    strip_include_prefix = "icicle/src",
    deps = [":hdrs"],
)

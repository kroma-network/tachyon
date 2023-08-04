load("@rules_cc//cc:defs.bzl", "cc_library")
load(":build_defs.bzl", "goldilocks_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "base_field",
    srcs = [
        "src/goldilocks_base_field.cpp",
    ],
    hdrs = [
        "src/goldilocks_base_field.hpp",
        "src/goldilocks_base_field_avx.hpp",
        "src/goldilocks_base_field_avx512.hpp",
        "src/goldilocks_base_field_batch.hpp",
        "src/goldilocks_base_field_scalar.hpp",
        "src/goldilocks_base_field_tools.hpp",
    ],
    copts = goldilocks_copts(),
    include_prefix = "third_party/polygon_zkevm_goldilocks/include",
    includes = ["src"],
    strip_include_prefix = "src",
    deps = ["@local_config_gmp//:gmp"],
)

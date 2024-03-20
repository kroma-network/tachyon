load("@kroma_network_tachyon//bazel:tachyon.bzl", "if_has_openmp")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rapidsnark",
    srcs = [
        "src/binfile_utils.cpp",
        "src/fileloader.cpp",
        "src/fullprover.cpp",
        "src/logger.cpp",
        "src/prover.cpp",
        "src/verifier.cpp",
        "src/wtns_utils.cpp",
        "src/zkey_utils.cpp",
    ],
    hdrs = [
        "src/binfile_utils.hpp",
        "src/fileloader.hpp",
        "src/fullprover.hpp",
        "src/groth16.cpp",
        "src/groth16.hpp",
        "src/logger.hpp",
        "src/logging.hpp",
        "src/prover.h",
        "src/random_generator.hpp",
        "src/verifier.h",
        "src/wtns_utils.hpp",
        "src/zkey_utils.hpp",
    ],
    copts = if_has_openmp(["-fopenmp"]),
    defines = ["NOZK"] + if_has_openmp(["USE_OPENMP"]),
    includes = ["src"],
    linkopts = if_has_openmp(["-fopenmp"]),
    deps = [
        "@iden3_ffiasm//c",
        "@nlohmann_json//:json",
    ],
)

cc_binary(
    name = "main_prover",
    srcs = ["src/main_prover.cpp"],
    deps = [":rapidsnark"],
)

cc_binary(
    name = "main_verifier",
    srcs = ["src/main_verifier.cpp"],
    deps = [":rapidsnark"],
)

cc_binary(
    name = "test_public_size",
    srcs = ["src/test_public_size.c"],
    deps = [":rapidsnark"],
)

cc_binary(
    name = "test_prover",
    srcs = ["src/test_prover.cpp"],
    deps = [":rapidsnark"],
)

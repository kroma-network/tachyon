load("@iden3_ffiasm//c:build_defs.bzl", "generate_prime_field")
load("@kroma_network_tachyon//bazel:tachyon.bzl", "if_has_openmp")

package(default_visibility = ["//visibility:public"])

generate_prime_field(
    name = "fr",
    modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617",
)

generate_prime_field(
    name = "fq",
    modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583",
)

cc_library(
    name = "c",
    srcs = [
        "alt_bn128.cpp",
        "misc.cpp",
        "naf.cpp",
        "splitparstr.cpp",
    ],
    hdrs = [
        "alt_bn128.hpp",
        "curve.cpp",
        "curve.hpp",
        "exp.hpp",
        "exp2.hpp",
        "f12field.cpp",
        "f12field.hpp",
        "f2field.cpp",
        "f2field.hpp",
        "f6field.cpp",
        "f6field.hpp",
        "fft.cpp",
        "fft.hpp",
        "misc.hpp",
        "multiexp.cpp",
        "multiexp.hpp",
        "naf.hpp",
        "splitparstr.hpp",
    ],
    copts = if_has_openmp(["-fopenmp"]),
    defines = if_has_openmp(["USE_OPENMP"]),
    linkopts = if_has_openmp(["-fopenmp"]),
    deps = [
        ":fq",
        ":fr",
    ],
)

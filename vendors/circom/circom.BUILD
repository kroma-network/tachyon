CURVES = [
    "bls12381",
    "bn128",
    "goldilocks",
    "grumpkin",
    "pallas",
    "secq256r1",
    "vesta",
]

TPLS = [
    "code_producers/src/c_elements/{}/fr.asm",
    "code_producers/src/c_elements/{}/fr.cpp",
    "code_producers/src/c_elements/{}/fr.hpp",
]

C_ELEMENTS = [tpl.format(curve) for tpl in TPLS for curve in CURVES]

filegroup(
    name = "generated_files",
    srcs = C_ELEMENTS + [
        "code_producers/src/c_elements/common/calcwit.cpp",
        "code_producers/src/c_elements/common/calcwit.hpp",
        "code_producers/src/c_elements/common/circom.hpp",
        "code_producers/src/c_elements/common/main.cpp",
    ],
    visibility = ["//visibility:public"],
)

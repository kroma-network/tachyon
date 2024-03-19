def witness_gen_library(
        name,
        gendep,
        prime = "bn128"):
    native.cc_library(
        name = name,
        srcs = [
            gendep,
            "//circomlib/generated/common:common_srcs",
        ],
        data = [gendep],
        deps = [
            "//circomlib/generated/common:common_hdrs",
            "//circomlib/generated/{}:fr".format(prime),
            "@com_google_absl//absl/container:flat_hash_map",
        ],
    )

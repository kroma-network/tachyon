def witness_gen_library(
        name,
        gendep,
        prime = "bn128"):
    native.cc_library(
        name = name,
        srcs = [
            gendep,
            "@kroma_network_circom//circomlib/generated/common:common_srcs",
        ],
        data = [gendep],
        deps = [
            "@com_google_absl//absl/container:flat_hash_map",
            "@kroma_network_circom//circomlib/generated/common:common_hdrs",
            "@kroma_network_circom//circomlib/generated/{}:fr".format(prime),
        ],
    )

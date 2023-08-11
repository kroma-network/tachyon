def goldilocks_copts():
    return select({
        "@kroma_network_tachyon//:x86_64_and_polygon_zkevm": [
            "-mavx",
            "-mavx512f",
        ],
        "//conditions:default": [],
    })

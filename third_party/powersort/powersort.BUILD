load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "powersort",
    hdrs = [
        "src/algorithms.h",
        "src/sorts/insertionsort.h",
        "src/sorts/merging.h",
        "src/sorts/powersort.h",
    ],
    include_prefix = "third_party/powersort/include",
    strip_include_prefix = "src",
)

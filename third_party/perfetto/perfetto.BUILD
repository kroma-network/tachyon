load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "perfetto",
    hdrs = ["sdk/perfetto.h"],
    srcs = ["sdk/perfetto.cc"],
    strip_include_prefix = "sdk",
    include_prefix = "third_party/perfetto",
)

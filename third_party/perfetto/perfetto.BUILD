load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "perfetto",
    srcs = ["sdk/perfetto.cc"],
    hdrs = ["sdk/perfetto.h"],
    include_prefix = "third_party/perfetto",
    strip_include_prefix = "sdk",
)

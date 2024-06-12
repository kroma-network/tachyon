load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

HEADERS = [
    "omp.h",
    "ompx.h",
    "omp-tools.h",
    "ompt.h",
]

# NOTE: This is only for macos.
#       On other platforms openmp should work without local config.
genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        "@kroma_network_tachyon//:macos_x86_64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /usr/local/include/$$file $(@D)/$$file
      done
    """,
        "@kroma_network_tachyon//:macos_aarch64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /opt/homebrew/opt/libomp/include/$$file $(@D)/$$file
      done
    """,
        "//conditions:default": "",
    }),
)

cc_library(
    name = "omp",
    hdrs = HEADERS,
    include_prefix = "third_party/omp/include",
    includes = ["."],
    linkopts = select({
        "@kroma_network_tachyon//:macos_x86_64": ["-L/usr/local/lib"],
        "@kroma_network_tachyon//:macos_aarch64": ["-L/opt/homebrew/opt/libomp/lib"],
        "//conditions:default": [],
    }) + [
        "-lomp",
    ],
)

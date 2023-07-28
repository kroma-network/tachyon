load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

HEADERS = [
    "gmp.h",
    "gmpxx.h",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        "@kroma_network_tachyon//:linux_x86_64": """
      mkdir -p $(@D)/
      ln -sf /usr/include/gmpxx.h $(@D)/gmpxx.h
      ln -sf {usr_include}/gmp.h $(@D)/gmp.h
    """,
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
        ln -sf /opt/homebrew/include/$$file $(@D)/$$file
      done
    """,
        "//conditions:default": "",
    }),
)

cc_library(
    name = "gmp",
    hdrs = HEADERS,
    include_prefix = "third_party/gmp/include",
    includes = ["."],
    linkopts = [
        "-lgmpxx",
        "-lgmp",
    ] + select({
        "@kroma_network_tachyon//:macos_x86_64": ["-L/usr/local/lib"],
        "@kroma_network_tachyon//:macos_aarch64": ["-L/opt/homebrew/lib"],
        "//conditions:default": [],
    }),
)

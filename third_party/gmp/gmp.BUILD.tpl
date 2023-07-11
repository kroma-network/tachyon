load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "macos_aarch64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
)

HEADERS = [
    "gmp.h",
    "gmpxx.h",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        ":linux_x86_64": """
      mkdir -p $(@D)/
      ln -sf /usr/include/gmpxx.h $(@D)/gmpxx.h
      ln -sf {usr_include}/gmp.h $(@D)/gmp.h
    """,
        ":macos_x86_64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /usr/local/include/$$file $(@D)/$$file
      done
    """,
        ":macos_aarch64": """
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
        ":macos_x86_64": ["-L/usr/local/lib"],
        ":macos_aarch64": ["-L/opt/homebrew/lib"],
        "//conditions:default": [],
    }),
)

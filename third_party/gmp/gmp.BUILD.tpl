load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "macos_x86",
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
    "gmp/gmp.h",
    "gmp/gmpxx.h",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        ":macos_x86": """
      mkdir -p $(@D)/gmp
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /usr/local/include/$$file $(@D)/gmp/$$file
      done
    """,
        ":macos_aarch64": """
      mkdir -p $(@D)/gmp
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /opt/homebrew/include/$$file $(@D)/gmp/$$file
      done
    """,
        "//conditions:default": "",
    }),
)

cc_library(
    name = "gmp",
    hdrs = select({
        "@platforms//os:macos": HEADERS,
        "//conditions:default": [],
    }),
    includes = ["gmp"],
    linkopts = [
        "-lgmpxx",
        "-lgmp",
    ] + select({
        ":macos_x86": ["-L/usr/local/lib"],
        ":macos_aarch64": ["-L/opt/homebrew/lib"],
        "//conditions:default": [],
    }),
)

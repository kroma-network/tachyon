load(
    "@rules_rust//rust:defs.bzl",
    "rust_binary",
    "rust_library",
    "rust_static_library",
    "rust_test",
)

def tachyon_rustc_flags():
    return select({
        "@platforms//os:macos": [
            "-C",
            "link-arg=-framework",
            "-C",
            "link-arg=Foundation",
        ],
        "//conditions:default": [],
    })

def tachyon_rust_tags():
    return ["rust"]

def tachyon_rust_binary(
        name,
        tags = [],
        rustc_flags = [],
        **kwargs):
    rust_binary(
        name = name,
        rustc_flags = rustc_flags + tachyon_rustc_flags(),
        tags = tags + tachyon_rust_tags(),
        **kwargs
    )

def tachyon_rust_library(
        name,
        tags = [],
        rustc_flags = [],
        **kwargs):
    rust_library(
        name = name,
        rustc_flags = rustc_flags + tachyon_rustc_flags(),
        tags = tags + tachyon_rust_tags(),
        **kwargs
    )

def tachyon_rust_static_library(
        name,
        tags = [],
        rustc_flags = [],
        **kwargs):
    rust_static_library(
        name = name,
        rustc_flags = rustc_flags + tachyon_rustc_flags(),
        tags = tags + tachyon_rust_tags(),
        **kwargs
    )

def tachyon_rust_test(
        name,
        tags = [],
        rustc_flags = [],
        **kwargs):
    rust_test(
        name = name,
        rustc_flags = rustc_flags + tachyon_rustc_flags(),
        tags = tags + tachyon_rust_tags(),
        **kwargs
    )

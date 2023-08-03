load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

def tachyon_rust_library(
        name,
        tags = [],
        **kwargs):
    rust_library(
        name = name,
        tags = tags + ["rust"],
        **kwargs
    )

def tachyon_rust_test(
        name,
        tags = [],
        **kwargs):
    rust_test(
        name = name,
        tags = tags + ["rust"],
        **kwargs
    )

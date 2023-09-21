load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pybind11_bazel//:build_defs.bzl", "PYBIND_COPTS", "PYBIND_DEPS", "PYBIND_FEATURES")
load(":tachyon_cc.bzl", "tachyon_cc_binary", "tachyon_cc_library")

def tachyon_pybind_extension(
        name,
        copts = [],
        features = [],
        linkopts = [],
        tags = [],
        deps = [],
        **kwargs):
    # Mark common dependencies as required for build_cleaner.
    tags = tags + ["req_dep=%s" % dep for dep in PYBIND_DEPS]

    tachyon_cc_binary(
        name = name + ".so",
        copts = copts + PYBIND_COPTS + select({
            "@pybind11//:msvc_compiler": [],
            "//conditions:default": [
                "-fvisibility=hidden",
            ],
        }),
        features = features + PYBIND_FEATURES,
        linkopts = linkopts + select({
            "@pybind11//:msvc_compiler": [],
            "@pybind11//:osx": ["-undefined", "dynamic_lookup"],
            "//conditions:default": ["-Wl,-Bsymbolic"],
        }),
        linkshared = True,
        tags = tags,
        deps = deps + PYBIND_DEPS,
        **kwargs
    )

def tachyon_pybind_library(
        name,
        copts = [],
        features = [],
        tags = [],
        deps = [],
        **kwargs):
    # Mark common dependencies as required for build_cleaner.
    tags = tags + ["req_dep=%s" % dep for dep in PYBIND_DEPS]

    tachyon_cc_library(
        name = name,
        copts = copts + PYBIND_COPTS,
        features = features + PYBIND_FEATURES,
        tags = tags,
        deps = deps + PYBIND_DEPS,
        **kwargs
    )

def tachyon_py_library(
        name,
        python_version = "PY3",
        srcs_version = "PY3",
        **kwargs):
    py_library(
        name = name,
        python_version = python_version,
        srcs_version = srcs_version,
        **kwargs
    )

def tachyon_py_binary(
        name,
        python_version = "PY3",
        srcs_version = "PY3",
        **kwargs):
    py_binary(
        name = name,
        python_version = python_version,
        srcs_version = srcs_version,
        **kwargs
    )

def tachyon_py_test(
        name,
        python_version = "PY3",
        srcs_version = "PY3",
        **kwargs):
    py_test(
        name = name,
        python_version = python_version,
        srcs_version = srcs_version,
        **kwargs
    )

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

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

def tachyon_py_unittest(
        name,
        **kwargs):
    py_test(
        name = name + "_unittests",
        **kwargs
    )

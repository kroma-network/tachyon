licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "numpy",
    srcs = ["tachyon_numpy_dummy.py"],
    srcs_version = "PY3",
)

alias(
    name = "headers",
    actual = "@local_config_python//:numpy_headers",
)

genrule(
    name = "dummy",
    outs = ["tachyon_numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)

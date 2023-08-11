package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "matplotlibcpp17",
    hdrs = glob(["include/matplotlibcpp17/*.h"]),
    include_prefix = "third_party/matplotlibcpp17/include",
    includes = ["include"],
    strip_include_prefix = "include/matplotlibcpp17",
    deps = [
        "@local_config_python//:numpy_headers",
        "@pybind11//:pybind11_embed",
    ],
)

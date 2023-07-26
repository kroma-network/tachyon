load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def pybind11_deps():
    if not native.existing_rule("pybind11_bazel"):
        http_archive(
            name = "pybind11_bazel",
            strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
            urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
        )

    if not native.existing_rule("pybind11"):
        http_archive(
            name = "pybind11",
            strip_prefix = "pybind11-2.11.1",
            urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
            build_file = "@pybind11_bazel//:pybind11.BUILD",
        )

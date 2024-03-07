load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def js_deps():
    if not native.existing_rule("aspect_bazel_lib"):
        http_archive(
            name = "aspect_bazel_lib",
            sha256 = "262e3d6693cdc16dd43880785cdae13c64e6a3f63f75b1993c716295093d117f",
            strip_prefix = "bazel-lib-1.38.1",
            url = "https://github.com/aspect-build/bazel-lib/releases/download/v1.38.1/bazel-lib-v1.38.1.tar.gz",
        )

    if not native.existing_rule("aspect_rules_js"):
        http_archive(
            name = "aspect_rules_js",
            sha256 = "d9ceb89e97bb5ad53b278148e01a77a3e9100db272ce4ebdcd59889d26b9076e",
            strip_prefix = "rules_js-1.34.0",
            url = "https://github.com/aspect-build/rules_js/releases/download/v1.34.0/rules_js-v1.34.0.tar.gz",
        )

    if not native.existing_rule("rules_nodejs"):
        http_archive(
            name = "rules_nodejs",
            sha256 = "a50986c7d2f2dc43a5b9b81a6245fd89bdc4866f1d5e316d9cef2782dd859292",
            strip_prefix = "rules_nodejs-6.0.5",
            url = "https://github.com/bazelbuild/rules_nodejs/releases/download/v6.0.5/rules_nodejs-v6.0.5.tar.gz",
        )

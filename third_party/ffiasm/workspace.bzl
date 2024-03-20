"""loads the ffiasm library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "iden3_ffiasm",
        urls = tf_mirror_urls("https://github.com/kroma-network/ffiasm/archive/3ea01c80d3e1709ba554d4d6d1b24bec101520c7.tar.gz"),
        sha256 = "9b570f5177e28793f9e96cb951bdb419f16d10787fe5f3af63cabfb89f376800",
        strip_prefix = "ffiasm-3ea01c80d3e1709ba554d4d6d1b24bec101520c7",
        link_files = {
            "//third_party/ffiasm:ffiasm.BUILD": "c/BUILD.bazel",
            "//third_party/ffiasm:build_defs.bzl": "c/build_defs.bzl",
        },
    )

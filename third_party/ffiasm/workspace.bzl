"""loads the ffiasm library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "iden3_ffiasm",
        urls = tf_mirror_urls("https://github.com/kroma-network/ffiasm/archive/7db5a442b12dba825ec35361441aa116dc3bff8b.tar.gz"),
        sha256 = "f46b13887bbf5c9f07b3b8eda577a95264408b496a16d96f70c7f3a899d4fa0c",
        strip_prefix = "ffiasm-7db5a442b12dba825ec35361441aa116dc3bff8b",
        link_files = {
            "//third_party/ffiasm:ffiasm.BUILD": "c/BUILD.bazel",
            "//third_party/ffiasm:build_defs.bzl": "c/build_defs.bzl",
        },
    )

"""loads the ffiasm library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "iden3_ffiasm",
        urls = tf_mirror_urls("https://github.com/kroma-network/ffiasm/archive/fd362c583c577c15b512e96d2683eeefc658f7f2.tar.gz"),
        sha256 = "50e147f50a9ffd84b9806158bca1d4428e4fc67ee56a78bcd1bdbf87934c3eb0",
        strip_prefix = "ffiasm-fd362c583c577c15b512e96d2683eeefc658f7f2",
    )

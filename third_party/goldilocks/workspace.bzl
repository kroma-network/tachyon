"""loads the goldilocks library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "goldilocks",
        urls = tf_mirror_urls("https://github.com/0xPolygonHermez/goldilocks/archive/e33eb7cb60ba483807040ee5e9ad6d1ea8dbf315.tar.gz"),
        sha256 = "675f31ee46f826047a4d9d7a0fa5d4b5f14caec9d43cb175be744e4e8da84fd4",
        strip_prefix = "goldilocks-e33eb7cb60ba483807040ee5e9ad6d1ea8dbf315",
        build_file = "//third_party/goldilocks:goldilocks.BUILD",
    )

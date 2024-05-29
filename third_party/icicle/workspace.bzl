"""loads the icicle library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "icicle",
        urls = tf_mirror_urls("https://github.com/ingonyama-zk/icicle/archive/v2.3.1.tar.gz"),
        sha256 = "f718adb20846dd704d6d83e96d1ac78debb4b7c435318950b4d2d791c22f8503",
        strip_prefix = "icicle-2.3.1",
        patch_file = [
            "@kroma_network_tachyon//third_party/icicle:rename.patch",
            "@kroma_network_tachyon//third_party/icicle:pragma.patch",
            "@kroma_network_tachyon//third_party/icicle:inlinize.patch",
        ],
        build_file = "//third_party/icicle:icicle.BUILD",
    )

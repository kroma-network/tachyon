"""loads the icicle library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "icicle",
        urls = tf_mirror_urls("https://github.com/ingonyama-zk/icicle/archive/355e45cc2a7454127c6d4c29b7b256831d99754d.tar.gz"),
        sha256 = "b803ea54948572ebb4666281512ab10cd51b7629f3fef7135c781f128ddfe2d9",
        strip_prefix = "icicle-355e45cc2a7454127c6d4c29b7b256831d99754d",
        patch_file = [
            "@kroma_network_tachyon//third_party/icicle:rename.patch",
            "@kroma_network_tachyon//third_party/icicle:rename-cu-cc.patch",
            "@kroma_network_tachyon//third_party/icicle:rename-cu-h.patch",
            "@kroma_network_tachyon//third_party/icicle:pragma.patch",
            "@kroma_network_tachyon//third_party/icicle:inlinize.patch",
            "@kroma_network_tachyon//third_party/icicle:remove-kernels-from-header.patch",
        ],
        build_file = "//third_party/icicle:icicle.BUILD",
    )

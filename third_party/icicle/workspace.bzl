"""loads the icicle library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "icicle",
        urls = tf_mirror_urls("https://github.com/ingonyama-zk/icicle/archive/4fef5423462a72a37fe66cee89e24eae083cc112.tar.gz"),
        sha256 = "ae88dccb706668aa2f323de53734b325f26e9995f8a73559ea1c87daf36c5b06",
        strip_prefix = "icicle-4fef5423462a72a37fe66cee89e24eae083cc112",
        patch_file = [
            "@kroma_network_tachyon//third_party/icicle:rename.patch",
            "@kroma_network_tachyon//third_party/icicle:pragma.patch",
            "@kroma_network_tachyon//third_party/icicle:inlinize.patch",
            "@kroma_network_tachyon//third_party/icicle:remove_kernels_from_header.patch",
        ],
        build_file = "//third_party/icicle:icicle.BUILD",
        link_files = {"//third_party/icicle:build_defs.bzl": "build_defs.bzl"},
    )

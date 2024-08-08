"""loads the icicle library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "icicle",
        urls = tf_mirror_urls("https://github.com/ingonyama-zk/icicle/archive/53f34aade57ff2b54a7eb1afadd563c143d4aa69.tar.gz"),
        sha256 = "eac59b94592d3409e83b47a414f77d67d80780a5cf1461e62c0d5213c9b10c50",
        strip_prefix = "icicle-53f34aade57ff2b54a7eb1afadd563c143d4aa69",
        patch_file = [
            "@kroma_network_tachyon//third_party/icicle:rename.patch",
            "@kroma_network_tachyon//third_party/icicle:pragma.patch",
            "@kroma_network_tachyon//third_party/icicle:inlinize.patch",
            "@kroma_network_tachyon//third_party/icicle:remove_kernels_from_header.patch",
            "@kroma_network_tachyon//third_party/icicle:separate_msm_config.patch",
            "@kroma_network_tachyon//third_party/icicle:separate_ntt_algorithm.patch",
        ],
        build_file = "//third_party/icicle:icicle.BUILD",
        link_files = {"//third_party/icicle:build_defs.bzl": "build_defs.bzl"},
    )

"""loads the rapidsnark library for benchmarking purposes."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "iden3_rapidsnark",
        urls = tf_mirror_urls("https://github.com/iden3/rapidsnark/archive/9fa3f0bc7a31ec791b937d233ef0b444d73eda8b.tar.gz"),
        sha256 = "7f03f95a1bbb6d83c0399f3b962d2b05b5a94574050d6c45c492acc3c07a363b",
        strip_prefix = "rapidsnark-9fa3f0bc7a31ec791b937d233ef0b444d73eda8b",
        build_file = "//third_party/rapidsnark:rapidsnark.BUILD",
        patch_file = ["@kroma_network_tachyon//third_party/rapidsnark:nozk.patch"],
    )

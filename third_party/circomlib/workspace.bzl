"""loads the circomlib library for testing purpose."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "kroma_network_circomlib",
        urls = tf_mirror_urls("https://github.com/kroma-network/circomlib/archive/b040af932308aa7b6e9d034cc43c18bea4ed1ab4.tar.gz"),
        sha256 = "446215fc03bb1d763dd0475d4f21ec269f65c9897a111de5b6272816c244911b",
        strip_prefix = "circomlib-b040af932308aa7b6e9d034cc43c18bea4ed1ab4",
    )

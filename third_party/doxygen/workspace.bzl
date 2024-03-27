"""loads the doxygen binary used for generating Tachyon API docs."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "doxygen_archive",
        build_file = "//third_party/doxygen:doxygen_archive.BUILD",
        urls = tf_mirror_urls("https://www.doxygen.nl/files/doxygen-1.10.0.linux.bin.tar.gz"),
        sha256 = "dcfc9aa4cc05aef1f0407817612ad9e9201d9bf2ce67cecf95a024bba7d39747",
        strip_prefix = "doxygen-1.10.0",
    )

"""loads the hwloc library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "pdqsort",
        urls = tf_mirror_urls("https://github.com/orlp/pdqsort/archive/b1ef26a55cdb60d236a5cb199c4234c704f46726.tar.gz"),
        sha256 = "1df2463f94ebd926f402e7bcd92bf4a16f7a35732080a607fe4716888f1edbb5",
        strip_prefix = "pdqsort-b1ef26a55cdb60d236a5cb199c4234c704f46726",
        build_file = "//third_party/pdqsort:pdqsort.BUILD",
    )

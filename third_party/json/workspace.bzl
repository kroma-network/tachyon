"""loads the json library used by rapidsnark."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "nlohmann_json",
        urls = tf_mirror_urls("https://github.com/nlohmann/json/archive/v3.11.3.tar.gz"),
        sha256 = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
        strip_prefix = "json-3.11.3",
    )

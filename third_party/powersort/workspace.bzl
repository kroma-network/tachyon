"""loads the powersort library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "powersort",
        urls = tf_mirror_urls("https://github.com/sebawild/powersort/archive/48e31e909280ca43bb2c33dd3df9922b0a0f3f84.tar.gz"),
        sha256 = "89122b7e7e2a0f0b41cc5411f9adde581769ff2f7d141335ce7e5011b932da06",
        strip_prefix = "powersort-48e31e909280ca43bb2c33dd3df9922b0a0f3f84",
        build_file = "//third_party/powersort:powersort.BUILD",
        patch_file = [
            "@kroma_network_tachyon//third_party/powersort:fix_sign_compare_warning.patch",
            "@kroma_network_tachyon//third_party/powersort:fix_multiple_definitions.patch",
            "@kroma_network_tachyon//third_party/powersort:fix_static_assertion.patch",
            # In c++ 17, std::binary_function is removed.
            # See https://en.cppreference.com/w/cpp/utility/functional/binary_function.
            "@kroma_network_tachyon//third_party/powersort:remove_binary_function.patch",
        ],
    )

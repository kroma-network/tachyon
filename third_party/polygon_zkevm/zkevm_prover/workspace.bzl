"""loads the hwloc library, used by Tachyon."""

load("//third_party:repo.bzl", "tachyon_http_archive", "tf_mirror_urls")

def repo():
    tachyon_http_archive(
        name = "polygon_zkevm_zkevm_prover",
        urls = tf_mirror_urls("https://github.com/0xPolygonHermez/zkevm-prover/archive/v2.0.1-hotfix.1.tar.gz"),
        sha256 = "27fdbee058ea483557601db5c8b257a2665e8ddfbb5055f503858f87439423b0",
        strip_prefix = "zkevm-prover-2.0.1-hotfix.1",
        build_file = "//third_party/polygon_zkevm/zkevm_prover:zkevm_prover.BUILD",
        link_files = {
            "//third_party/polygon_zkevm/zkevm_prover:fr_fail.h": "src/ffiasm/fr_fail.h",
            "//third_party/polygon_zkevm/zkevm_prover:fr_fail.cc": "src/ffiasm/fr_fail.cc",
        },
    )

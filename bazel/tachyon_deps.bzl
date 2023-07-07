load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/env:env_configure.bzl", "env_configure")
load("//third_party/gmp:gmp_configure.bzl", "gmp_configure")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")

def tachyon_deps():
    cuda_configure(name = "local_config_cuda")
    env_configure(name = "local_config_env")
    gmp_configure(name = "local_config_gmp")
    rocm_configure(name = "local_config_rocm")

    if not native.existing_rule("bazel_skylib"):
        http_archive(
            name = "bazel_skylib",
            sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
            urls = [
                "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
                "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            ],
        )

    # Needed by com_github_google_glog
    if not native.existing_rule("com_github_gflags_gflags"):
        http_archive(
            name = "com_github_gflags_gflags",
            strip_prefix = "gflags-2.2.2",
            urls = [
                "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
                "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
            ],
        )

    if not native.existing_rule("com_github_google_glog"):
        http_archive(
            name = "com_github_google_glog",
            sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
            strip_prefix = "glog-0.6.0",
            urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
        )

    if not native.existing_rule("com_google_absl"):
        http_archive(
            name = "com_google_absl",
            sha256 = "51d676b6846440210da48899e4df618a357e6e44ecde7106f1e44ea16ae8adc7",
            strip_prefix = "abseil-cpp-20230125.3",
            urls = ["https://github.com/abseil/abseil-cpp/archive/20230125.3.zip"],
            patch_args = ["-p1"],
            patches = ["@kroma_network_tachyon//third_party/absl:add_missing_linkopts.patch"],
        )

    if not native.existing_rule("com_google_googletest"):
        http_archive(
            name = "com_google_googletest",
            sha256 = "ffa17fbc5953900994e2deec164bb8949879ea09b411e07f215bfbb1f87f4632",
            strip_prefix = "googletest-1.13.0",
            urls = ["https://github.com/google/googletest/archive/v1.13.0.zip"],
        )

    # Needed by com_google_googletest
    if not native.existing_rule("com_googlesource_code_re2"):
        http_archive(
            name = "com_googlesource_code_re2",
            sha256 = "4ccdd5aafaa1bcc24181e6dd3581c3eee0354734bb9f3cb4306273ffa434b94f",
            strip_prefix = "re2-2023-06-02",
            urls = ["https://github.com/google/re2/archive/2023-06-02.tar.gz"],
        )

    # Needed by com_google_googletest
    if not native.existing_rule("rules_python"):
        http_archive(
            name = "rules_python",
            sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
            strip_prefix = "rules_python-0.23.1",
            urls = ["https://github.com/bazelbuild/rules_python/releases/download/0.23.1/rules_python-0.23.1.tar.gz"],
        )

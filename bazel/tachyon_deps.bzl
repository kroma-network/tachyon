load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/circomlib:workspace.bzl", circomlib = "repo")
load("//third_party/doxygen:workspace.bzl", doxygen = "repo")
load("//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("//third_party/env:env_configure.bzl", "env_configure")
load("//third_party/ffiasm:workspace.bzl", ffiasm = "repo")
load("//third_party/gmp:gmp_configure.bzl", "gmp_configure")
load("//third_party/goldilocks:workspace.bzl", goldilocks = "repo")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/icicle:workspace.bzl", icicle = "repo")
load("//third_party/json:workspace.bzl", json = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/node_addon_api:install_node_addon_api.bzl", "install_node_addon_api")
load("//third_party/omp:omp_configure.bzl", "omp_configure")
load("//third_party/pdqsort:workspace.bzl", pdqsort = "repo")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/rapidsnark:workspace.bzl", rapidsnark = "repo")

def tachyon_deps():
    cuda_configure(name = "local_config_cuda")
    env_configure(name = "local_config_env")
    gmp_configure(name = "local_config_gmp")
    omp_configure(name = "local_config_omp")
    rocm_configure(name = "local_config_rocm")
    python_configure(name = "local_config_python")

    circomlib()
    doxygen()
    eigen3()
    ffiasm()
    goldilocks()
    hwloc()
    icicle()
    json()
    nasm()
    pdqsort()
    rapidsnark()

    install_node_addon_api(name = "node_addon_api")

    if not native.existing_rule("bazel_skylib"):
        http_archive(
            name = "bazel_skylib",
            sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
            urls = [
                "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
                "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            ],
        )

    if not native.existing_rule("com_github_google_benchmark"):
        http_archive(
            name = "com_github_google_benchmark",
            sha256 = "2aab2980d0376137f969d92848fbb68216abb07633034534fc8c65cc4e7a0e93",
            strip_prefix = "benchmark-1.8.2",
            urls = ["https://github.com/google/benchmark/archive/v1.8.2.tar.gz"],
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
        # TODO(chokobole): Bump up to 0.6.0.
        # glog v0.6.0 can't be built with --config cuda.
        http_archive(
            name = "com_github_google_glog",
            sha256 = "21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a",
            strip_prefix = "glog-0.5.0",
            urls = ["https://github.com/google/glog/archive/v0.5.0.zip"],
            patch_args = ["-p1"],
            patches = [
                "@kroma_network_tachyon//third_party/glog:enable_constexpr_check_op.patch",
                "@kroma_network_tachyon//third_party/glog:enable_elaborate_ostream_operator.patch",
            ],
        )

    if not native.existing_rule("com_github_soblin_matplotlibcpp17"):
        http_archive(
            name = "com_github_soblin_matplotlibcpp17",
            sha256 = "f5889241b57c6a06e9f07745c2e4062bb2c584a67756aad3dd042a65147cc830",
            strip_prefix = "matplotlibcpp17-779f108f192a27761cae3724929850fe65de625c",
            urls = ["https://github.com/soblin/matplotlibcpp17/archive/779f108f192a27761cae3724929850fe65de625c.tar.gz"],
            build_file = "@kroma_network_tachyon//third_party/matplotlibcpp17:matplotlibcpp17.BUILD",
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

    if not native.existing_rule("com_google_boringssl"):
        http_archive(
            name = "com_google_boringssl",
            sha256 = "c17af0c3ddd72613a6cacf3761cebaa583806fc49fb53ac021ae479aa8265c93",
            strip_prefix = "boringssl-b69f4d27a75dcf4b94138790883b44274fab56c2",
            urls = ["https://github.com/google/boringssl/archive/b69f4d27a75dcf4b94138790883b44274fab56c2.tar.gz"],
            patch_args = ["-p1"],
            patches = ["@kroma_network_tachyon//third_party/boringssl:blake2b512.patch"],
        )

    if not native.existing_rule("com_google_googletest"):
        http_archive(
            name = "com_google_googletest",
            sha256 = "ffa17fbc5953900994e2deec164bb8949879ea09b411e07f215bfbb1f87f4632",
            strip_prefix = "googletest-1.13.0",
            urls = ["https://github.com/google/googletest/archive/v1.13.0.zip"],
            patch_args = ["-p1"],
            patches = [
                "@kroma_network_tachyon//third_party/gtest:add_missing_linkopts.patch",
                "@kroma_network_tachyon//third_party/gtest:print_with_to_string.patch",
            ],
        )

    if not native.existing_rule("com_github_tencent_rapidjson"):
        http_archive(
            name = "com_github_tencent_rapidjson",
            strip_prefix = "rapidjson-1.1.0",
            sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
            urls = ["https://github.com/Tencent/rapidjson/archive/v1.1.0.zip"],
            build_file = "@kroma_network_tachyon//third_party:rapidjson/rapidjson.BUILD",
        )

    # Needed by com_google_googletest
    if not native.existing_rule("com_googlesource_code_re2"):
        http_archive(
            name = "com_googlesource_code_re2",
            sha256 = "4ccdd5aafaa1bcc24181e6dd3581c3eee0354734bb9f3cb4306273ffa434b94f",
            strip_prefix = "re2-2023-06-02",
            urls = ["https://github.com/google/re2/archive/2023-06-02.tar.gz"],
        )

    if not native.existing_rule("cxx.rs"):
        http_archive(
            name = "cxx.rs",
            sha256 = "ab918028d85b1c0a5c21a24d9a78bb6a1adc78608191e6ff382bcbb2bcd57ccd",
            strip_prefix = "cxx-1.0.102",
            urls = ["https://github.com/dtolnay/cxx/archive/refs/tags/1.0.102.tar.gz"],
            patch_args = ["-p1"],
            patches = ["@kroma_network_tachyon//third_party/cxx_rs:add_more_args_to_cxx_bridge.patch"],
        )

    if not native.existing_rule("rules_pkg"):
        http_archive(
            name = "rules_pkg",
            sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
            urls = ["https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz"],
        )

    # Needed by com_google_googletest
    if not native.existing_rule("rules_python"):
        http_archive(
            name = "rules_python",
            sha256 = "84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841",
            strip_prefix = "rules_python-0.23.1",
            urls = ["https://github.com/bazelbuild/rules_python/releases/download/0.23.1/rules_python-0.23.1.tar.gz"],
        )

    if not native.existing_rule("rules_rust"):
        http_archive(
            name = "rules_rust",
            sha256 = "9d04e658878d23f4b00163a72da3db03ddb451273eb347df7d7c50838d698f49",
            urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.26.0/rules_rust-v0.26.0.tar.gz"],
        )

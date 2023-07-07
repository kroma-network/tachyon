""" Downloads clang and configures the crosstool using bazel's autoconf."""

load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_autoconf_impl")
load(":download_clang.bzl", "download_clang")

_TACHYON_DOWNLOAD_CLANG = "TACHYON_DOWNLOAD_CLANG"
_TACHYON_NEED_CUDA = "TACHYON_NEED_CUDA"

def _cc_clang_autoconf(repo_ctx):
    if repo_ctx.os.environ.get(_TACHYON_DOWNLOAD_CLANG) != "1":
        return
    if repo_ctx.os.environ.get(_TACHYON_NEED_CUDA) == "1":
        # Clang is handled separately for CUDA configs.
        # See cuda_configure.bzl for more details.
        return

    download_clang(repo_ctx, out_folder = "extra_tools")
    overridden_tools = {"gcc": "extra_tools/bin/clang"}
    cc_autoconf_impl(repo_ctx, overridden_tools)

cc_download_clang_toolchain = repository_rule(
    environ = [
        _TACHYON_DOWNLOAD_CLANG,
        _TACHYON_NEED_CUDA,
    ],
    implementation = _cc_clang_autoconf,
)

def _gmp_configure_impl(repository_ctx):
    repository_ctx.template(
        "third_party/gmp/gmp.BUILD",
        Label("//third_party/gmp:gmp.BUILD.tpl"),
    )
    repository_ctx.symlink("third_party/gmp/gmp.BUILD", "BUILD.bazel")

gmp_configure = repository_rule(
    implementation = _gmp_configure_impl,
)

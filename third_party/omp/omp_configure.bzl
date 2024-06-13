def _omp_configure_impl(repository_ctx):
    repository_ctx.symlink(Label("//third_party/omp:omp.BUILD"), "BUILD.bazel")

omp_configure = repository_rule(
    implementation = _omp_configure_impl,
)

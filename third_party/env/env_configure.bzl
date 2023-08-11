def _env_autoconf_impl(repository_ctx):
    repository_ctx.symlink(Label("//third_party/env:BUILD.bazel"), "BUILD.bazel")
    project_root = repository_ctx.path(Label("//:BUILD.bazel")).dirname

    repository_ctx.template(
        "env.bzl",
        Label("//third_party/env:env.bzl.tpl"),
        {
            "%{PROJECT_ROOT}": "\"%s\"" % project_root,
        },
    )

env_configure = repository_rule(
    implementation = _env_autoconf_impl,
)

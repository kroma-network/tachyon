load(
    "//third_party/remote_config:common.bzl",
    "get_bash_bin",
)

def _fail(msg):
    """Output failure message when auto configuration fails."""
    fail("node-addon-api install failed: %s\n" % (msg))

def _install_node_addon_api_impl(repository_ctx):
    repository_ctx.symlink(Label("//third_party/node_addon_api:package.json"), "package.json")

    bash_bin = get_bash_bin(repository_ctx)

    cmd = [bash_bin, "-c", "npm install"]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
        _fail("Failed npm install: %s." % result.stderr)

    cmd = [bash_bin, "-c", "npx node-gyp install --devdir ."]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
        _fail("Failed node-gyp install: %s." % result.stderr)

    cmd = [bash_bin, "-c", "node --version"]
    result = repository_ctx.execute(cmd)
    if result.return_code != 0:
        _fail("Failed node --version: %s." % result.stderr)

    version = result.stdout.strip()
    if version.startswith("v"):
        version = version[1:]

    repository_ctx.template(
        "BUILD.bazel",
        Label("//third_party/node_addon_api:BUILD.tpl"),
        {
            "%{NODE_VERSION}": version,
        },
    )

install_node_addon_api = repository_rule(
    implementation = _install_node_addon_api_impl,
)

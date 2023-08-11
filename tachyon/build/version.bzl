load("//tachyon/build:lastchange.bzl", "LastChangeInfo")

def _write_version_header_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)

    arguments = [
        output.path,
        ctx.attr.project,
        str(ctx.attr.major),
        str(ctx.attr.minor),
        str(ctx.attr.patch),
    ]
    if len(ctx.attr.prerelease) > 0:
        arguments.append("--prerelease=%s" % (ctx.attr.prerelease))
    if ctx.attr.lastchange != None:
        lastchanges = ctx.attr.lastchange[LastChangeInfo].lastchange.to_list()
        arguments.append("--lastchange=%s" % (lastchanges[0].path))
    ctx.actions.run(
        inputs = ctx.files.lastchange,
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [output],
        progress_message = "Generating %s" % (output.short_path),
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([output]))]

write_version_header = rule(
    implementation = _write_version_header_impl,
    attrs = {
        "_tool": attr.label(
            cfg = "exec",
            allow_single_file = True,
            executable = True,
            default = Label("@kroma_network_tachyon//tachyon/build:write_version_header.py"),
        ),
        "output": attr.string(mandatory = True),
        "project": attr.string(mandatory = True),
        "major": attr.int(mandatory = True),
        "minor": attr.int(mandatory = True),
        "patch": attr.int(mandatory = True),
        "prerelease": attr.string(default = ""),
        "lastchange": attr.label(providers = [LastChangeInfo]),
    },
)

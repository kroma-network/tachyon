load("//tachyon/build:lastchange.bzl", "LastChangeInfo")

def _write_version_header_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    tool_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/build:write_version_header.py)", [ctx.attr._tool])

    command = "%s %s %s %s %s %s" % (
        tool_path,
        output.path,
        ctx.attr.project,
        ctx.attr.major,
        ctx.attr.minor,
        ctx.attr.patch,
    )
    if len(ctx.attr.prerelease) > 0:
        command += " --prerelease %s" % (ctx.attr.prerelease)
    if ctx.attr.lastchange != None:
        lastchanges = ctx.attr.lastchange[LastChangeInfo].lastchange.to_list()
        command += " --lastchange %s" % (lastchanges[0].path)
    ctx.actions.run_shell(
        inputs = ctx.files.lastchange,
        tools = ctx.files._tool,
        outputs = [output],
        progress_message = "Generating %s" % (output.short_path),
        command = command,
    )

    return [DefaultInfo(files = depset([output]))]

write_version_header = rule(
    implementation = _write_version_header_impl,
    attrs = {
        "_tool": attr.label(
            allow_single_file = True,
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

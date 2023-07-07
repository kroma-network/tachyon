load("@local_config_env//:env.bzl", "PROJECT_ROOT")

LastChangeInfo = provider(
    "Info needed to extract last revision and commit time.",
    fields = ["lastchange"],
)

def _lastchange_impl(ctx):
    outputs = [ctx.actions.declare_file("LASTCHANGE"), ctx.actions.declare_file("LASTCHANGE.committime")]
    tool_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/build:lastchange.py)", [ctx.attr._tool])
    ctx.actions.run_shell(
        tools = ctx.files._tool,
        outputs = outputs,
        execution_requirements = {
            "no-cache": "1",
            "no-remote": "1",
        },
        progress_message = "Generating LASTCHANGE",
        command = "%s --source-dir %s --output LASTCHANGE && mv LASTCHANGE LASTCHANGE.committime %s" % (
            tool_path,
            PROJECT_ROOT,
            outputs[0].dirname,
        ),
    )

    return [
        DefaultInfo(files = depset(outputs)),
        LastChangeInfo(lastchange = depset(outputs)),
    ]

lastchange = rule(
    implementation = _lastchange_impl,
    attrs = {
        "_tool": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/build:lastchange.py"),
        ),
    },
)

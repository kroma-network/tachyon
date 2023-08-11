load("@local_config_env//:env.bzl", "PROJECT_ROOT")

LastChangeInfo = provider(
    "Info needed to extract last revision and commit time.",
    fields = ["lastchange"],
)

def _lastchange_impl(ctx):
    outputs = [ctx.actions.declare_file("LASTCHANGE"), ctx.actions.declare_file("LASTCHANGE.committime")]
    ctx.actions.run_shell(
        tools = [ctx.executable._tool],
        outputs = outputs,
        progress_message = "Generating LASTCHANGE",
        command = "%s --source-dir %s --output LASTCHANGE && mv LASTCHANGE LASTCHANGE.committime %s" % (
            ctx.executable._tool.path,
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
            cfg = "exec",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/build:lastchange.py"),
        ),
    },
)

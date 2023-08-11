def get_var(ctx, name):
    return name in ctx.var and ctx.var[name]

def attrs():
    return {
        "out": attr.output(mandatory = True),
        "_tool": attr.label(
            cfg = "exec",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/build:write_buildflag_header.py"),
        ),
    }

def gen_buildflag_header_helper(ctx, flags):
    content = "--flags " + " ".join(flags)
    definition = "%s.definition" % (ctx.attr.name)
    out = ctx.outputs.out
    header_guard = out.path[len(ctx.genfiles_dir.path) + 1:]

    ctx.actions.run_shell(
        tools = [ctx.executable._tool],
        outputs = [out],
        progress_message = "Generating buildflag %s" % (out.short_path),
        command = "echo '%s' > %s &&  %s --header-guard %s --rulename %s --definition %s --output %s --gen-dir %s && rm %s" % (
            content,
            definition,
            ctx.executable._tool.path,
            header_guard,  # --header-guard
            ctx.build_file_path,  # --rulename
            definition,  # --defintion
            out.basename,  # --output
            out.dirname,  #  --gen-dir
            definition,
        ),
    )

    return [DefaultInfo(files = depset([out]))]

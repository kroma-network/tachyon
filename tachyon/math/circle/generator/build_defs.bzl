load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_circle_point_impl(ctx):
    cpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/circle/generator:cpu.h.tpl)", [ctx.attr.cpu_hdr_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--base_field=%s" % (ctx.attr.base_field),
        "--base_field_hdr=%s" % (ctx.attr.base_field_hdr),
        "--scalar_field=%s" % (ctx.attr.scalar_field),
        "--scalar_field_hdr=%s" % (ctx.attr.scalar_field_hdr),
        "-x=%s" % (ctx.attr.x),
        "-y=%s" % (ctx.attr.y),
        "--cpu_hdr_tpl_path=%s" % (cpu_hdr_tpl_path),
    ]
    if len(ctx.attr.class_name) > 0:
        arguments.append("--class=%s" % (ctx.attr.class_name))

    ctx.actions.run(
        inputs = [ctx.files.cpu_hdr_tpl_path[0]],
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_circle_point = rule(
    implementation = _generate_circle_point_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(),
        "base_field": attr.string(mandatory = True),
        "base_field_hdr": attr.string(mandatory = True),
        "scalar_field": attr.string(mandatory = True),
        "scalar_field_hdr": attr.string(mandatory = True),
        "x": attr.string(mandatory = True),
        "y": attr.string(mandatory = True),
        "cpu_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/circle/generator:cpu.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/circle/generator"),
        ),
    },
)

def generate_circle_points(
        name,
        namespace,
        base_field,
        base_field_hdr,
        base_field_dep,
        scalar_field,
        scalar_field_hdr,
        scalar_field_dep,
        x,
        y,
        class_name = "",
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_circle_point(
            namespace = namespace,
            class_name = class_name,
            base_field = base_field,
            base_field_hdr = base_field_hdr,
            scalar_field = scalar_field,
            scalar_field_hdr = scalar_field_hdr,
            x = x,
            y = y,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = [
            base_field_dep,
            scalar_field_dep,
            "//tachyon/math/circle:circle",
            "//tachyon/math/circle:circle_point",
        ],
        **kwargs
    )

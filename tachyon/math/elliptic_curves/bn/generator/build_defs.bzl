load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_bn_curve_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--fq12_hdr=%s" % (ctx.attr.fq12_hdr),
        "--g1_hdr=%s" % (ctx.attr.g1_hdr),
        "--g2_hdr=%s" % (ctx.attr.g2_hdr),
        "-x=%s" % (ctx.attr.x),
        "--twist_type=%s" % (ctx.attr.twist_type),
    ]
    if len(ctx.attr.class_name) > 0:
        arguments.append("--class=%s" % (ctx.attr.class_name))

    for twist_mul_by_q_x in ctx.attr.twist_mul_by_q_x:
        arguments.append("--twist_mul_by_q_x=%s" % (twist_mul_by_q_x))

    for twist_mul_by_q_y in ctx.attr.twist_mul_by_q_y:
        arguments.append("--twist_mul_by_q_y=%s" % (twist_mul_by_q_y))

    ctx.actions.run(
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_bn_curve = rule(
    implementation = _generate_bn_curve_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(),
        "fq12_hdr": attr.string(mandatory = True),
        "g1_hdr": attr.string(mandatory = True),
        "g2_hdr": attr.string(mandatory = True),
        "x": attr.string(mandatory = True),
        "twist_mul_by_q_x": attr.string_list(mandatory = True),
        "twist_mul_by_q_y": attr.string_list(mandatory = True),
        "twist_type": attr.string(mandatory = True, values = ["M", "D"]),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/elliptic_curves/bn/generator"),
        ),
    },
)

def generate_bn_curves(
        name,
        namespace,
        fq12_hdr,
        fq12_dep,
        g1_hdr,
        g1_dep,
        g2_hdr,
        g2_dep,
        x,
        twist_mul_by_q_x,
        twist_mul_by_q_y,
        twist_type,
        class_name = "",
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_bn_curve(
            namespace = namespace,
            class_name = class_name,
            fq12_hdr = fq12_hdr,
            g1_hdr = g1_hdr,
            g2_hdr = g2_hdr,
            x = x,
            twist_mul_by_q_x = twist_mul_by_q_x,
            twist_mul_by_q_y = twist_mul_by_q_y,
            twist_type = twist_type,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = [
            fq12_dep,
            g1_dep,
            g2_dep,
            "//tachyon/base:logging",
            "//tachyon/math/elliptic_curves/bn:bn_curve",
            "//tachyon/math/elliptic_curves/pairing:twist_type",
        ],
        **kwargs
    )

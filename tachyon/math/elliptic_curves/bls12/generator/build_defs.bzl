load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_bls12_curve_impl(ctx):
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

    ctx.actions.run(
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_bls12_curve = rule(
    implementation = _generate_bls12_curve_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(),
        "fq12_hdr": attr.string(mandatory = True),
        "g1_hdr": attr.string(mandatory = True),
        "g2_hdr": attr.string(mandatory = True),
        "x": attr.string(mandatory = True),
        "twist_type": attr.string(mandatory = True, values = ["M", "D"]),
        "_tool": attr.label(
            # TODO(chokobole): Change it to "exec" we can build it on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/elliptic_curves/bls12/generator"),
        ),
    },
)

def generate_bls12_curves(
        name,
        namespace,
        fq12_hdr,
        fq12_dep,
        g1_hdr,
        g1_dep,
        g2_hdr,
        g2_dep,
        x,
        twist_type,
        class_name = "",
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_bls12_curve(
            namespace = namespace,
            class_name = class_name,
            fq12_hdr = fq12_hdr,
            g1_hdr = g1_hdr,
            g2_hdr = g2_hdr,
            x = x,
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
            "//tachyon/math/elliptic_curves/bls12:bls12_curve",
            "//tachyon/math/elliptic_curves/pairing:twist_type",
        ],
        **kwargs
    )

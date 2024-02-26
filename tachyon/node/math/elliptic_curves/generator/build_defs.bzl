load("//bazel:tachyon.bzl", "if_node_binding")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ec_point_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--type=%s" % (ctx.attr.type),
    ]

    ctx.actions.run(
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_ec_point = rule(
    implementation = _generate_ec_point_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "type": attr.string(mandatory = True),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/node/math/elliptic_curves/generator"),
        ),
    },
)

def generate_ec_points(
        name,
        g1_deps):
    for n in [
        ("gen_fq_hdr", "fq.h"),
        ("gen_fq_src", "fq.cc"),
        ("gen_fr_hdr", "fr.h"),
        ("gen_fr_src", "fr.cc"),
        ("gen_g1_hdr", "g1.h"),
        ("gen_g1_src", "g1.cc"),
    ]:
        generate_ec_point(
            type = name,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = "fq",
        hdrs = ["fq.h"],
        srcs = if_node_binding(["fq.cc"]),
        deps = g1_deps + [
            "//tachyon/node/math/finite_fields:prime_field",
        ],
    )

    tachyon_cc_library(
        name = "fr",
        hdrs = ["fr.h"],
        srcs = if_node_binding(["fr.cc"]),
        deps = g1_deps + [
            "//tachyon/node/math/finite_fields:prime_field",
        ],
    )

    tachyon_cc_library(
        name = "g1",
        hdrs = ["g1.h"],
        srcs = if_node_binding(["g1.cc"]),
        deps = g1_deps + [
            "//tachyon/node/math/elliptic_curves/short_weierstrass:points",
        ],
    )

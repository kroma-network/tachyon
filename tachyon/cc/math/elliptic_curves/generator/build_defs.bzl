load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ec_point_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--type=%s" % (ctx.attr.type),
        "--fq_limb_nums=%s" % (ctx.attr.fq_limb_nums),
        "--fr_limb_nums=%s" % (ctx.attr.fr_limb_nums),
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
        "fq_limb_nums": attr.int(mandatory = True),
        "fr_limb_nums": attr.int(mandatory = True),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/cc/math/elliptic_curves/generator"),
        ),
    },
)

def generate_ec_points(
        name,
        fq_limb_nums,
        fr_limb_nums,
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
            fq_limb_nums = fq_limb_nums,
            fr_limb_nums = fr_limb_nums,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = "fq",
        hdrs = ["fq.h"],
        srcs = ["fq.cc"],
        deps = g1_deps + [
            "//tachyon/cc:export",
        ],
    )

    tachyon_cc_library(
        name = "fr",
        hdrs = ["fr.h"],
        srcs = ["fr.cc"],
        deps = g1_deps + [
            "//tachyon/cc:export",
        ],
    )

    tachyon_cc_library(
        name = "g1",
        hdrs = ["g1.h"],
        srcs = ["g1.cc"],
        deps = g1_deps + [
            ":fq",
            ":fr",
        ],
    )

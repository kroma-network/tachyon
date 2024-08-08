load("//bazel:tachyon.bzl", "if_gpu_is_configured")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library", "tachyon_cuda_library")

def _generate_ec_point_impl(ctx):
    point_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point.h.tpl)", [ctx.attr.point_hdr_tpl_path])
    point_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point.cc.tpl)", [ctx.attr.point_src_tpl_path])
    point_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point_traits.h.tpl)", [ctx.attr.point_traits_hdr_tpl_path])
    point_type_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point_type_traits.h.tpl)", [ctx.attr.point_type_traits_hdr_tpl_path])
    msm_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.h.tpl)", [ctx.attr.msm_hdr_tpl_path])
    msm_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.cc.tpl)", [ctx.attr.msm_src_tpl_path])
    msm_gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.h.tpl)", [ctx.attr.msm_gpu_hdr_tpl_path])
    msm_gpu_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.cc.tpl)", [ctx.attr.msm_gpu_src_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--type=%s" % (ctx.attr.type),
        "--point_hdr_tpl_path=%s" % (point_hdr_tpl_path),
        "--point_src_tpl_path=%s" % (point_src_tpl_path),
        "--point_traits_hdr_tpl_path=%s" % (point_traits_hdr_tpl_path),
        "--point_type_traits_hdr_tpl_path=%s" % (point_type_traits_hdr_tpl_path),
        "--msm_hdr_tpl_path=%s" % (msm_hdr_tpl_path),
        "--msm_src_tpl_path=%s" % (msm_src_tpl_path),
        "--msm_gpu_hdr_tpl_path=%s" % (msm_gpu_hdr_tpl_path),
        "--msm_gpu_src_tpl_path=%s" % (msm_gpu_src_tpl_path),
    ]

    ctx.actions.run(
        inputs = [
            ctx.files.point_hdr_tpl_path[0],
            ctx.files.point_src_tpl_path[0],
            ctx.files.point_traits_hdr_tpl_path[0],
            ctx.files.point_type_traits_hdr_tpl_path[0],
            ctx.files.msm_hdr_tpl_path[0],
            ctx.files.msm_src_tpl_path[0],
            ctx.files.msm_gpu_hdr_tpl_path[0],
            ctx.files.msm_gpu_src_tpl_path[0],
        ],
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
        "point_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point.h.tpl"),
        ),
        "point_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point.cc.tpl"),
        ),
        "point_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point_traits.h.tpl"),
        ),
        "point_type_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:point_type_traits.h.tpl"),
        ),
        "msm_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.h.tpl"),
        ),
        "msm_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.cc.tpl"),
        ),
        "msm_gpu_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.h.tpl"),
        ),
        "msm_gpu_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.cc.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator"),
        ),
    },
)

def generate_ec_points(
        name,
        g1_deps,
        g2_deps,
        g1_gpu_deps):
    for n in [
        ("gen_g1_hdr", "g1.h"),
        ("gen_g1_src", "g1.cc"),
        ("gen_g2_hdr", "g2.h"),
        ("gen_g2_src", "g2.cc"),
        ("gen_g1_point_traits", "g1_point_traits.h"),
        ("gen_g2_point_traits", "g2_point_traits.h"),
        ("gen_g1_point_type_traits", "g1_point_type_traits.h"),
        ("gen_g2_point_type_traits", "g2_point_type_traits.h"),
        ("gen_msm_hdr", "msm.h"),
        ("gen_msm_src", "msm.cc"),
        ("gen_msm_gpu_hdr", "msm_gpu.h"),
        ("gen_msm_gpu_src", "msm_gpu.cc"),
    ]:
        generate_ec_point(
            type = name,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = "g1",
        hdrs = [
            "g1.h",
            "g1_point_traits.h",
            "g1_point_type_traits.h",
        ],
        srcs = ["g1.cc"],
        deps = g1_deps + [
            "//tachyon/c/math/elliptic_curves:point_traits_forward",
        ],
    )

    tachyon_cc_library(
        name = "g2",
        hdrs = [
            "g2.h",
            "g2_point_traits.h",
            "g2_point_type_traits.h",
        ],
        srcs = ["g2.cc"],
        deps = g2_deps + [
            "//tachyon/c/math/elliptic_curves:point_traits_forward",
        ],
    )

    tachyon_cc_library(
        name = "msm",
        hdrs = ["msm.h"],
        srcs = ["msm.cc"],
        deps = [
            ":g1",
            "//tachyon/c/math/elliptic_curves/msm",
        ],
    )

    tachyon_cuda_library(
        name = "msm_gpu",
        hdrs = ["msm_gpu.h"],
        srcs = if_gpu_is_configured(["msm_gpu.cc"]),
        deps = g1_gpu_deps + [
            ":g1",
            "//tachyon/c/math/elliptic_curves/msm:msm_gpu",
        ],
    )

    tachyon_cc_library(
        name = name,
        deps = [
            ":msm",
            ":msm_gpu",
        ],
    )

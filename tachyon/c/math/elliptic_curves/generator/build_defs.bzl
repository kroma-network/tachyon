load("//bazel:tachyon.bzl", "if_gpu_is_configured")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library", "tachyon_cuda_library")

def _generate_ec_point_impl(ctx):
    prime_field_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.h.tpl)", [ctx.attr.prime_field_hdr_tpl_path])
    prime_field_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.cc.tpl)", [ctx.attr.prime_field_src_tpl_path])
    prime_field_type_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field_type_traits.h.tpl)", [ctx.attr.prime_field_type_traits_hdr_tpl_path])
    ext_field_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field.h.tpl)", [ctx.attr.ext_field_hdr_tpl_path])
    ext_field_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field.cc.tpl)", [ctx.attr.ext_field_src_tpl_path])
    ext_field_type_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field_type_traits.h.tpl)", [ctx.attr.ext_field_type_traits_hdr_tpl_path])
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
        "--fq_limb_nums=%s" % (ctx.attr.fq_limb_nums),
        "--fr_limb_nums=%s" % (ctx.attr.fr_limb_nums),
        "--degree=%s" % (ctx.attr.degree),
        "--base_field_degree=%s" % (ctx.attr.base_field_degree),
        "--has_specialized_g1_msm_kernels=%s" % (ctx.attr.has_specialized_g1_msm_kernels),
        "--prime_field_hdr_tpl_path=%s" % (prime_field_hdr_tpl_path),
        "--prime_field_src_tpl_path=%s" % (prime_field_src_tpl_path),
        "--prime_field_type_traits_hdr_tpl_path=%s" % (prime_field_type_traits_hdr_tpl_path),
        "--ext_field_hdr_tpl_path=%s" % (ext_field_hdr_tpl_path),
        "--ext_field_src_tpl_path=%s" % (ext_field_src_tpl_path),
        "--ext_field_type_traits_hdr_tpl_path=%s" % (ext_field_type_traits_hdr_tpl_path),
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
            ctx.files.prime_field_hdr_tpl_path[0],
            ctx.files.prime_field_src_tpl_path[0],
            ctx.files.prime_field_type_traits_hdr_tpl_path[0],
            ctx.files.ext_field_hdr_tpl_path[0],
            ctx.files.ext_field_src_tpl_path[0],
            ctx.files.ext_field_type_traits_hdr_tpl_path[0],
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
        "fq_limb_nums": attr.int(mandatory = True),
        "fr_limb_nums": attr.int(mandatory = True),
        "degree": attr.int(mandatory = True),
        "base_field_degree": attr.int(mandatory = True),
        "has_specialized_g1_msm_kernels": attr.bool(mandatory = True),
        "prime_field_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.h.tpl"),
        ),
        "prime_field_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.cc.tpl"),
        ),
        "prime_field_type_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field_type_traits.h.tpl"),
        ),
        "ext_field_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field.h.tpl"),
        ),
        "ext_field_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field.cc.tpl"),
        ),
        "ext_field_type_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:ext_field_type_traits.h.tpl"),
        ),
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
        fq_limb_nums,
        fr_limb_nums,
        fq2_deps,
        fq6_deps,
        fq12_deps,
        g1_deps,
        g2_deps,
        g1_gpu_deps,
        g1_msm_kernels_deps = []):
    for n in [
        ("gen_fq_hdr", "fq.h", 0, 0),
        ("gen_fq_src", "fq.cc", 0, 0),
        ("gen_fr_hdr", "fr.h", 0, 0),
        ("gen_fr_src", "fr.cc", 0, 0),
        ("gen_fq2_hdr", "fq2.h", 2, 1),
        ("gen_fq2_src", "fq2.cc", 2, 1),
        ("gen_fq3_hdr", "fq3.h", 3, 1),
        ("gen_fq6_hdr", "fq6.h", 6, 2),
        ("gen_fq6_src", "fq6.cc", 6, 2),
        ("gen_fq12_hdr", "fq12.h", 12, 6),
        ("gen_fq12_src", "fq12.cc", 12, 6),
        ("gen_fq2_type_traits_hdr", "fq2_type_traits.h", 2, 1),
        ("gen_fq6_type_traits_hdr", "fq6_type_traits.h", 6, 2),
        ("gen_fq12_type_traits_hdr", "fq12_type_traits.h", 12, 6),
        ("gen_g1_hdr", "g1.h", 0, 0),
        ("gen_g1_src", "g1.cc", 0, 0),
        ("gen_g2_hdr", "g2.h", 0, 0),
        ("gen_g2_src", "g2.cc", 0, 0),
        ("gen_fq_type_traits", "fq_type_traits.h", 0, 0),
        ("gen_fr_type_traits", "fr_type_traits.h", 0, 0),
        ("gen_g1_point_traits", "g1_point_traits.h", 0, 0),
        ("gen_g2_point_traits", "g2_point_traits.h", 0, 0),
        ("gen_g1_point_type_traits", "g1_point_type_traits.h", 0, 0),
        ("gen_g2_point_type_traits", "g2_point_type_traits.h", 0, 0),
        ("gen_msm_hdr", "msm.h", 0, 0),
        ("gen_msm_src", "msm.cc", 0, 0),
        ("gen_msm_gpu_hdr", "msm_gpu.h", 0, 0),
        ("gen_msm_gpu_src", "msm_gpu.cc", 0, 0),
    ]:
        generate_ec_point(
            type = name,
            fq_limb_nums = fq_limb_nums,
            fr_limb_nums = fr_limb_nums,
            name = n[0],
            out = n[1],
            degree = n[2],
            base_field_degree = n[3],
            has_specialized_g1_msm_kernels = len(g1_msm_kernels_deps) > 0,
        )

    tachyon_cc_library(
        name = "fq",
        hdrs = [
            "fq.h",
            "fq_type_traits.h",
        ],
        srcs = ["fq.cc"],
        deps = g1_deps + [
            "//tachyon/c:export",
            "//tachyon/c/base:type_traits_forward",
        ],
    )

    tachyon_cc_library(
        name = "fr",
        hdrs = [
            "fr.h",
            "fr_type_traits.h",
        ],
        srcs = ["fr.cc"],
        deps = g1_deps + [
            "//tachyon/c:export",
            "//tachyon/c/base:type_traits_forward",
        ],
    )

    tachyon_cc_library(
        name = "fq2",
        hdrs = [
            "fq2.h",
            "fq2_type_traits.h",
        ],
        srcs = ["fq2.cc"],
        deps = fq2_deps + [":fq"],
    )

    tachyon_cc_library(
        name = "fq6",
        hdrs = [
            "fq6.h",
            "fq6_type_traits.h",
        ],
        srcs = ["fq6.cc"],
        deps = fq6_deps + [":fq2"],
    )

    tachyon_cc_library(
        name = "fq12",
        hdrs = [
            "fq12.h",
            "fq12_type_traits.h",
        ],
        srcs = ["fq12.cc"],
        deps = fq12_deps + [":fq6"],
    )

    tachyon_cc_library(
        name = "g1",
        hdrs = [
            "g1.h",
            "g1_point_traits.h",
            "g1_point_type_traits.h",
        ],
        srcs = ["g1.cc"],
        deps = [
            ":fq",
            ":fr",
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
            ":fq2",
            ":fr",
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

    if name != "bls12_381":
        # NOTE(chokobole): bls12_381 scalar field causes a compliation error at PrimeFieldGpu::MulInPlace().
        tachyon_cuda_library(
            name = "msm_gpu",
            hdrs = ["msm_gpu.h"],
            srcs = if_gpu_is_configured(["msm_gpu.cc"]),
            deps = g1_gpu_deps + g1_msm_kernels_deps + [
                ":g1",
                "//tachyon/c/math/elliptic_curves/msm:msm_gpu",
            ],
        )

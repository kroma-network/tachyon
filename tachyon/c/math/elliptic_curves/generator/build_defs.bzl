load("//bazel:tachyon.bzl", "if_gpu_is_configured")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library", "tachyon_cuda_library")

def _generate_ec_point_impl(ctx):
    prime_field_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.h.tpl)", [ctx.attr.prime_field_hdr_tpl_path])
    prime_field_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.cc.tpl)", [ctx.attr.prime_field_src_tpl_path])
    prime_field_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field_traits.h.tpl)", [ctx.attr.prime_field_traits_hdr_tpl_path])
    g1_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1.h.tpl)", [ctx.attr.g1_hdr_tpl_path])
    g1_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1.cc.tpl)", [ctx.attr.g1_src_tpl_path])
    g1_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1_traits.h.tpl)", [ctx.attr.g1_traits_hdr_tpl_path])
    msm_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.h.tpl)", [ctx.attr.msm_hdr_tpl_path])
    msm_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm.cc.tpl)", [ctx.attr.msm_src_tpl_path])
    msm_gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.h.tpl)", [ctx.attr.msm_gpu_hdr_tpl_path])
    msm_gpu_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:msm_gpu.cc.tpl)", [ctx.attr.msm_gpu_src_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--type=%s" % (ctx.attr.type),
        "--fq_limb_nums=%s" % (ctx.attr.fq_limb_nums),
        "--fr_limb_nums=%s" % (ctx.attr.fr_limb_nums),
        "--has_specialized_g1_msm_kernels=%s" % (ctx.attr.has_specialized_g1_msm_kernels),
        "--prime_field_hdr_tpl_path=%s" % (prime_field_hdr_tpl_path),
        "--prime_field_src_tpl_path=%s" % (prime_field_src_tpl_path),
        "--prime_field_traits_hdr_tpl_path=%s" % (prime_field_traits_hdr_tpl_path),
        "--g1_hdr_tpl_path=%s" % (g1_hdr_tpl_path),
        "--g1_src_tpl_path=%s" % (g1_src_tpl_path),
        "--g1_traits_hdr_tpl_path=%s" % (g1_traits_hdr_tpl_path),
        "--msm_hdr_tpl_path=%s" % (msm_hdr_tpl_path),
        "--msm_src_tpl_path=%s" % (msm_src_tpl_path),
        "--msm_gpu_hdr_tpl_path=%s" % (msm_gpu_hdr_tpl_path),
        "--msm_gpu_src_tpl_path=%s" % (msm_gpu_src_tpl_path),
    ]

    ctx.actions.run(
        inputs = [
            ctx.files.prime_field_hdr_tpl_path[0],
            ctx.files.prime_field_src_tpl_path[0],
            ctx.files.prime_field_traits_hdr_tpl_path[0],
            ctx.files.g1_hdr_tpl_path[0],
            ctx.files.g1_src_tpl_path[0],
            ctx.files.g1_traits_hdr_tpl_path[0],
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
        "has_specialized_g1_msm_kernels": attr.bool(mandatory = True),
        "prime_field_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.h.tpl"),
        ),
        "prime_field_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field.cc.tpl"),
        ),
        "prime_field_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:prime_field_traits.h.tpl"),
        ),
        "g1_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1.h.tpl"),
        ),
        "g1_src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1.cc.tpl"),
        ),
        "g1_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/elliptic_curves/generator:g1_traits.h.tpl"),
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
        g1_deps,
        g1_gpu_deps,
        g1_msm_kernels_deps = []):
    for n in [
        ("gen_fq_hdr", "fq.h"),
        ("gen_fq_src", "fq.cc"),
        ("gen_fr_hdr", "fr.h"),
        ("gen_fr_src", "fr.cc"),
        ("gen_g1_hdr", "g1.h"),
        ("gen_g1_src", "g1.cc"),
        ("gen_fq_prime_field_traits", "fq_prime_field_traits.h"),
        ("gen_fr_prime_field_traits", "fr_prime_field_traits.h"),
        ("gen_g1_point_traits", "g1_point_traits.h"),
        ("gen_msm_hdr", "msm.h"),
        ("gen_msm_src", "msm.cc"),
        ("gen_msm_gpu_hdr", "msm_gpu.h"),
        ("gen_msm_gpu_src", "msm_gpu.cc"),
    ]:
        generate_ec_point(
            type = name,
            fq_limb_nums = fq_limb_nums,
            fr_limb_nums = fr_limb_nums,
            name = n[0],
            out = n[1],
            has_specialized_g1_msm_kernels = len(g1_msm_kernels_deps) > 0,
        )

    tachyon_cc_library(
        name = "fq",
        hdrs = [
            "fq.h",
            "fq_prime_field_traits.h",
        ],
        srcs = ["fq.cc"],
        deps = g1_deps + [
            "//tachyon/c:export",
            "//tachyon/cc/math/finite_fields:prime_field_conversions",
        ],
    )

    tachyon_cc_library(
        name = "fr",
        hdrs = [
            "fr.h",
            "fr_prime_field_traits.h",
        ],
        srcs = ["fr.cc"],
        deps = g1_deps + [
            "//tachyon/c:export",
            "//tachyon/cc/math/finite_fields:prime_field_conversions",
        ],
    )

    tachyon_cc_library(
        name = "g1",
        hdrs = [
            "g1.h",
            "g1_point_traits.h",
        ],
        srcs = ["g1.cc"],
        deps = [
            ":fq",
            ":fr",
            "//tachyon/cc/math/elliptic_curves:point_conversions",
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

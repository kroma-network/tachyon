load("//bazel:tachyon.bzl", "if_gpu_is_configured")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library", "tachyon_cuda_library")

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
            # TODO(chokobole): Change it to "exec" we can build it on macos.
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
        g1_gpu_deps):
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
            deps = g1_gpu_deps + [
                ":g1",
                "//tachyon/c/math/elliptic_curves/msm:msm_gpu",
            ],
        )

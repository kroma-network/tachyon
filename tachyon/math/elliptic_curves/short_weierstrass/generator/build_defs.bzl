load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ec_point_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "-a=%s" % (ctx.attr.a),
        "-b=%s" % (ctx.attr.b),
        "-x=%s" % (ctx.attr.x),
        "-y=%s" % (ctx.attr.y),
        "--fq_modulus=%s" % (ctx.attr.fq_modulus),
    ]
    if len(ctx.attr.class_name) > 0:
        arguments.append("--class=%s" % (ctx.attr.class_name))

    if len(ctx.attr.glv_coeffs) > 0:
        arguments.extend([
            "--endomorphism_coefficient=%s" % (ctx.attr.endomorphism_coefficient),
            "--lambda=%s" % (ctx.attr.lambda_),
            "--fr_modulus=%s" % (ctx.attr.fr_modulus),
        ])
        for i in range(len(ctx.attr.glv_coeffs)):
            arguments.append("--glv_coefficients=%s" % (ctx.attr.glv_coeffs[i]))

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
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "a": attr.string(mandatory = True),
        "b": attr.string(mandatory = True),
        "x": attr.string(mandatory = True),
        "y": attr.string(mandatory = True),
        "fq_modulus": attr.string(mandatory = True),
        "fr_modulus": attr.string(mandatory = True),
        "endomorphism_coefficient": attr.string(),
        "lambda_": attr.string(),
        "glv_coeffs": attr.string_list(),
        "_tool": attr.label(
            # TODO(chokobole): Change it to "exec" we can build it on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator"),
        ),
    },
)

def generate_ec_points(
        name,
        namespace,
        class_name,
        a,
        b,
        x,
        y,
        fq_modulus,
        fr_modulus,
        endomorphism_coefficient = "",
        lambda_ = "",
        glv_coeffs = [],
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_ec_point(
            namespace = namespace,
            class_name = class_name,
            a = a,
            b = b,
            x = x,
            y = y,
            fq_modulus = fq_modulus,
            fr_modulus = fr_modulus,
            endomorphism_coefficient = endomorphism_coefficient,
            lambda_ = lambda_,
            glv_coeffs = glv_coeffs,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = [
            ":fq",
            ":fr",
            "//tachyon/math/elliptic_curves/short_weierstrass:points",
            "//tachyon/math/elliptic_curves/short_weierstrass:sw_curve",
        ],
        **kwargs
    )

    tachyon_cc_library(
        name = "{}_gpu".format(name),
        hdrs = [":{}_gen_gpu_hdr".format(name)],
        deps = [
            ":{}".format(name),
            ":fq_gpu",
            ":fr_gpu",
            "//tachyon/math/elliptic_curves/short_weierstrass:sw_curve_gpu",
        ],
        **kwargs
    )

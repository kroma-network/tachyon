load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ec_point_impl(ctx):
    out = ctx.outputs.out
    tool_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator)", [ctx.attr._tool])

    cmd = "%s --out %s --namespace %s -a %s -b %s -x %s -y %s --fq_modulus %s" % (
        tool_path,
        out.path,
        ctx.attr.namespace,
        ctx.attr.a,
        ctx.attr.b,
        ctx.attr.x,
        ctx.attr.y,
        ctx.attr.fq_modulus,
    )

    if len(ctx.attr.class_name) > 0:
        cmd += " --class %s" % (ctx.attr.class_name)

    if len(ctx.attr.glv_coeffs) > 0:
        cmd += " --endomorphism_coefficient %s --lambda %s --fr_modulus %s" % (
            ctx.attr.endomorphism_coefficient,
            ctx.attr.lambda_,
            ctx.attr.fr_modulus,
        )
        for i in range(len(ctx.attr.glv_coeffs)):
            cmd += " --glv_coefficients %s" % (ctx.attr.glv_coeffs[i])

    ctx.actions.run_shell(
        tools = ctx.files._tool,
        outputs = [out],
        command = cmd,
    )

    return [DefaultInfo(files = depset([out]))]

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
        name = "{}_gen_hdr".format(name),
        out = "{}.h".format(name),
    )

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
        name = "{}_gen_cuda_hdr".format(name),
        out = "{}_cuda.cu.h".format(name),
    )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = [
            ":fq",
            ":fr",
            "//tachyon/math/elliptic_curves/short_weierstrass:points",
        ],
        **kwargs
    )

    tachyon_cc_library(
        name = "{}_cuda".format(name),
        hdrs = [":{}_gen_cuda_hdr".format(name)],
        deps = [
            ":{}".format(name),
            ":fq_cuda",
            ":fr_cuda",
            "//tachyon/math/elliptic_curves/short_weierstrass:sw_curve_cuda",
        ],
        **kwargs
    )

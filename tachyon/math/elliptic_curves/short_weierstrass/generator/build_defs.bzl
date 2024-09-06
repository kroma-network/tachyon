load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ec_point_impl(ctx):
    cpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator:cpu.h.tpl)", [ctx.attr.cpu_hdr_tpl_path])
    gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator:gpu.h.tpl)", [ctx.attr.gpu_hdr_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--base_field=%s" % (ctx.attr.base_field),
        "--base_field_degree=%s" % (ctx.attr.base_field_degree),
        "--base_field_hdr=%s" % (ctx.attr.base_field_hdr),
        "--scalar_field=%s" % (ctx.attr.scalar_field),
        "--scalar_field_hdr=%s" % (ctx.attr.scalar_field_hdr),
        "--cpu_hdr_tpl_path=%s" % (cpu_hdr_tpl_path),
        "--gpu_hdr_tpl_path=%s" % (gpu_hdr_tpl_path),
    ]
    if len(ctx.attr.class_name) > 0:
        arguments.append("--class=%s" % (ctx.attr.class_name))

    for a in ctx.attr.a:
        arguments.append("-a=%s" % (a))

    for b in ctx.attr.b:
        arguments.append("-b=%s" % (b))

    for x in ctx.attr.x:
        arguments.append("-x=%s" % (x))

    for y in ctx.attr.y:
        arguments.append("-y=%s" % (y))

    if len(ctx.attr.mul_by_a_override) > 0:
        arguments.append("--mul_by_a_override=%s" % (ctx.attr.mul_by_a_override))

    if len(ctx.attr.glv_coeffs) > 0:
        for endomorphism_coefficient in ctx.attr.endomorphism_coefficient:
            arguments.append("--endomorphism_coefficient=%s" % (endomorphism_coefficient))

        arguments.append("--lambda=%s" % (ctx.attr.lambda_))

        for glv_coeff in ctx.attr.glv_coeffs:
            arguments.append("--glv_coefficients=%s" % glv_coeff)

    ctx.actions.run(
        inputs = [ctx.files.cpu_hdr_tpl_path[0], ctx.files.gpu_hdr_tpl_path[0]],
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
        "class_name": attr.string(),
        "base_field_degree": attr.int(mandatory = True),
        "base_field": attr.string(mandatory = True),
        "base_field_hdr": attr.string(mandatory = True),
        "scalar_field": attr.string(mandatory = True),
        "scalar_field_hdr": attr.string(mandatory = True),
        "a": attr.string_list(mandatory = True),
        "b": attr.string_list(mandatory = True),
        "x": attr.string_list(mandatory = True),
        "y": attr.string_list(mandatory = True),
        "mul_by_a_override": attr.string(),
        "endomorphism_coefficient": attr.string_list(),
        "lambda_": attr.string(),
        "glv_coeffs": attr.string_list(),
        "cpu_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator:cpu.h.tpl"),
        ),
        "gpu_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/elliptic_curves/short_weierstrass/generator:gpu.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
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
        base_field_degree,
        base_field,
        base_field_hdr,
        base_field_dep,
        scalar_field,
        scalar_field_hdr,
        scalar_field_dep,
        a,
        b,
        x,
        y,
        class_name = "",
        mul_by_a_override = "",
        endomorphism_coefficient = [],
        lambda_ = "",
        glv_coeffs = [],
        gen_gpu = False,
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_ec_point(
            namespace = namespace,
            class_name = class_name,
            base_field_degree = base_field_degree,
            base_field = base_field,
            base_field_hdr = base_field_hdr,
            scalar_field = scalar_field,
            scalar_field_hdr = scalar_field_hdr,
            a = a,
            b = b,
            x = x,
            y = y,
            mul_by_a_override = mul_by_a_override,
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
            base_field_dep,
            scalar_field_dep,
            "//tachyon/math/elliptic_curves/short_weierstrass:points",
            "//tachyon/math/elliptic_curves/short_weierstrass:sw_curve",
            "@com_google_absl//absl/base",
        ],
        **kwargs
    )

    # TODO(chokobole): Remove this if condition once GPU G2 is implemented.
    if gen_gpu:
        tachyon_cc_library(
            name = "{}_gpu".format(name),
            hdrs = [":{}_gen_gpu_hdr".format(name)],
            deps = [
                ":{}".format(name),
                base_field_dep + "_gpu",
                scalar_field_dep + "_gpu",
            ],
            **kwargs
        )

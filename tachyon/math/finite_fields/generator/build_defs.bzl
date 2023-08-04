load("//bazel:tachyon.bzl", "if_gmp_backend")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_prime_field_impl(ctx):
    out = ctx.outputs.out
    tool_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator)", [ctx.attr._tool])

    cmd = "%s --out %s --namespace %s --modulus %s" % (
        tool_path,
        out.path,
        ctx.attr.namespace,
        ctx.attr.modulus,
    )

    if len(ctx.attr.class_name) > 0:
        cmd += " --class %s" % (ctx.attr.class_name)

    if len(ctx.attr.hdr_include_override):
        cmd += " --hdr_include_override '%s'" % (ctx.attr.hdr_include_override)

    if len(ctx.attr.special_prime_override):
        cmd += " --special_prime_override '%s'" % (ctx.attr.special_prime_override)

    ctx.actions.run_shell(
        tools = ctx.files._tool,
        outputs = [out],
        command = cmd,
    )

    return [DefaultInfo(files = depset([out]))]

generate_prime_field = rule(
    implementation = _generate_prime_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "modulus": attr.string(mandatory = True),
        "hdr_include_override": attr.string(),
        "special_prime_override": attr.string(),
        "_tool": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator"),
        ),
    },
)

def generate_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        hdr_include_override = "",
        special_prime_override = "",
        deps = [],
        **kwargs):
    generate_prime_field(
        namespace = namespace,
        class_name = class_name,
        modulus = modulus,
        name = "{}_gen_cc".format(name),
        out = "{}.cc".format(name),
    )

    generate_prime_field(
        namespace = namespace,
        class_name = class_name,
        modulus = modulus,
        hdr_include_override = hdr_include_override,
        special_prime_override = special_prime_override,
        name = "{}_gen_hdr".format(name),
        out = "{}.h".format(name),
    )

    generate_prime_field(
        namespace = namespace,
        class_name = class_name,
        modulus = modulus,
        name = "{}_gen_cuda_hdr".format(name),
        out = "{}_cuda.cu.h".format(name),
    )

    tachyon_cc_library(
        name = name,
        srcs = [":{}_gen_cc".format(name)],
        hdrs = [":{}_gen_hdr".format(name)],
        deps = deps + [
            "//tachyon/math/finite_fields:prime_field",
        ] + if_gmp_backend([
            "//tachyon/math/finite_fields:prime_field_gmp",
        ]),
        **kwargs
    )

    tachyon_cc_library(
        name = "{}_cuda".format(name),
        hdrs = [":{}_gen_cuda_hdr".format(name)],
        deps = [
            ":{}".format(name),
            "//tachyon/math/finite_fields:prime_field_cuda",
        ],
        **kwargs
    )

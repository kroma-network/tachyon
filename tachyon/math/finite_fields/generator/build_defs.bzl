load("//bazel:tachyon.bzl", "if_gmp_backend")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_prime_field_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--modulus=%s" % (ctx.attr.modulus),
    ]

    if len(ctx.attr.class_name) > 0:
        arguments.append("--class=%s" % (ctx.attr.class_name))

    if len(ctx.attr.hdr_include_override):
        arguments.append("--hdr_include_override=%s" % (ctx.attr.hdr_include_override))

    if len(ctx.attr.special_prime_override):
        arguments.append("--special_prime_override=%s" % (ctx.attr.special_prime_override))

    ctx.actions.run(
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

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
            # TODO(chokobole): Change it to "exec" we can build it on macos.
            cfg = "target",
            executable = True,
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
    for n in [
        ("{}_gen_cc".format(name), "{}.cc".format(name)),
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            hdr_include_override = hdr_include_override,
            special_prime_override = special_prime_override,
            name = n[0],
            out = n[1],
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
        name = "{}_gpu".format(name),
        hdrs = [":{}_gen_gpu_hdr".format(name)],
        deps = [
            ":{}".format(name),
            "//tachyon/math/finite_fields:prime_field_gpu",
        ],
        **kwargs
    )

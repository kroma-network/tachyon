load("//bazel:tachyon.bzl", "if_gmp_backend")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_prime_field_impl(ctx):
    out = ctx.outputs.out
    tool_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator)", [ctx.attr._tool])

    ctx.actions.run_shell(
        tools = ctx.files._tool,
        outputs = [out],
        command = "%s --out %s --namespace %s --class %s --modulus %s" % (
            tool_path,
            out.path,
            ctx.attr.namespace,
            ctx.attr.class_name,
            ctx.attr.modulus,
        ),
    )

    return [DefaultInfo(files = depset([out]))]

generate_prime_field = rule(
    implementation = _generate_prime_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "modulus": attr.string(mandatory = True),
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
        deps = [
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

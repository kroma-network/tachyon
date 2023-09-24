load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ext_prime_field_impl(ctx):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--class=%s" % (ctx.attr.class_name),
        "--degree=%s" % (ctx.attr.degree),
        "--non_residue=%s" % (ctx.attr.non_residue),
        "--base_field_hdr=%s" % (ctx.attr.base_field_hdr),
        "--base_field=%s" % (ctx.attr.base_field),
    ]

    ctx.actions.run(
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_ext_prime_field = rule(
    implementation = _generate_ext_prime_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "degree": attr.int(mandatory = True),
        "non_residue": attr.int(mandatory = True),
        "base_field_hdr": attr.string(mandatory = True),
        "base_field": attr.string(mandatory = True),
        "_tool": attr.label(
            # TODO(chokobole): Change it to "exec" we can build it on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/ext_prime_field_generator"),
        ),
    },
)

def _generate_ext_prime_fields(
        name,
        namespace,
        class_name,
        degree,
        non_residue,
        base_field_hdr = "",
        base_field = "",
        ext_prime_field_deps = [],
        deps = [],
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_ext_prime_field(
            namespace = namespace,
            class_name = class_name,
            degree = degree,
            non_residue = non_residue,
            base_field_hdr = base_field_hdr,
            base_field = base_field,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = deps + ext_prime_field_deps,
        **kwargs
    )

def generate_fp2s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 2,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp2"],
        **kwargs
    )

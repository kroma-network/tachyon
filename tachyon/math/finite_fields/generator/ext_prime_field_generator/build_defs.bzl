load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ext_prime_field_impl(ctx):
    fq_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/ext_prime_field_generator:fq.h.tpl)", [ctx.attr.fq_hdr_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--class=%s" % (ctx.attr.class_name),
        "--degree=%s" % (ctx.attr.degree),
        "--base_field_degree=%s" % (ctx.attr.base_field_degree),
        "--base_field_hdr=%s" % (ctx.attr.base_field_hdr),
        "--base_field=%s" % (ctx.attr.base_field),
        "--fq_hdr_tpl_path=%s" % (fq_hdr_tpl_path),
    ]

    for non_residue in ctx.attr.non_residue:
        arguments.append("--non_residue=%s" % (non_residue))

    if len(ctx.attr.mul_by_non_residue_override) > 0:
        arguments.append("--mul_by_non_residue_override=%s" % (ctx.attr.mul_by_non_residue_override))

    ctx.actions.run(
        inputs = [ctx.files.fq_hdr_tpl_path[0]],
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
        "non_residue": attr.string_list(mandatory = True),
        "base_field_degree": attr.int(mandatory = True),
        "base_field_hdr": attr.string(mandatory = True),
        "base_field": attr.string(mandatory = True),
        "mul_by_non_residue_override": attr.string(),
        "fq_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/ext_prime_field_generator:fq.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
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
        base_field_degree,
        base_field_hdr,
        base_field,
        mul_by_non_residue_override = "",
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
            base_field_degree = base_field_degree,
            base_field_hdr = base_field_hdr,
            base_field = base_field,
            mul_by_non_residue_override = mul_by_non_residue_override,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = deps + ext_prime_field_deps + [
            "//tachyon/base:logging",
        ],
        **kwargs
    )

def generate_fp2s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 2,
        base_field_degree = 1,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp2"],
        **kwargs
    )

def generate_fp3s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 3,
        base_field_degree = 1,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp3"],
        **kwargs
    )

def generate_fp4s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 4,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp4"],
        **kwargs
    )

def generate_fp6s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 6,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp6"],
        **kwargs
    )

def generate_fp12s(
        name,
        **kwargs):
    _generate_ext_prime_fields(
        name = name,
        degree = 12,
        base_field_degree = 6,
        ext_prime_field_deps = ["//tachyon/math/finite_fields:fp12"],
        **kwargs
    )

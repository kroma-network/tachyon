load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_binary_field_impl(ctx):
    binary_config_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_config.h.tpl)", [ctx.attr.binary_config_hdr_tpl])
    binary_cpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_cpu.h.tpl)", [ctx.attr.binary_cpu_hdr_tpl])
    binary_gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_gpu.h.tpl)", [ctx.attr.binary_gpu_hdr_tpl])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--class=%s" % (ctx.attr.class_name),
        "--modulus=%s" % (ctx.attr.modulus),
        "--binary_config_hdr_tpl_path=%s" % (binary_config_hdr_tpl_path),
        "--binary_cpu_hdr_tpl_path=%s" % (binary_cpu_hdr_tpl_path),
        "--binary_gpu_hdr_tpl_path=%s" % (binary_gpu_hdr_tpl_path),
    ]

    ctx.actions.run(
        inputs = [
            ctx.files.binary_config_hdr_tpl[0],
            ctx.files.binary_cpu_hdr_tpl[0],
            ctx.files.binary_gpu_hdr_tpl[0],
        ],
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_binary_field = rule(
    implementation = _generate_binary_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "modulus": attr.string(mandatory = True),
        "binary_config_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_config.h.tpl"),
        ),
        "binary_cpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_cpu.h.tpl"),
        ),
        "binary_gpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator:binary_field_gpu.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/binary_field_generator"),
        ),
    },
)

def generate_binary_fields(
        name,
        namespace,
        class_name,
        modulus,
        **kwargs):
    for n in [
        ("{}_gen_config_hdr".format(name), "{}_config.h".format(name)),
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_binary_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = "{}_config".format(name),
        hdrs = [":{}_gen_config_hdr".format(name)],
        deps = ["//tachyon:export"],
    )

    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = [
            ":{}_config".format(name),
            "//tachyon/math/finite_fields:binary_field",
        ],
        **kwargs
    )

    tachyon_cc_library(
        name = "{}_gpu".format(name),
        hdrs = [":{}_gen_gpu_hdr".format(name)],
        deps = [
            ":{}_config".format(name),
            "//tachyon/math/finite_fields:binary_field",
        ],
        **kwargs
    )

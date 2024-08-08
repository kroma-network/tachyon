load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_ext_field_impl(ctx):
    hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field.h.tpl)", [ctx.attr.hdr_tpl_path])
    src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field.cc.tpl)", [ctx.attr.src_tpl_path])
    type_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field_type_traits.h.tpl)", [ctx.attr.type_traits_hdr_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--class_name=%s" % (ctx.attr.class_name),
        "--base_field_class_name=%s" % (ctx.attr.base_field_class_name),
        "--display_name=%s" % (ctx.attr.display_name),
        "--base_field_display_name=%s" % (ctx.attr.base_field_display_name),
        "--c_base_field_hdr=%s" % (ctx.attr.c_base_field_hdr),
        "--native_type=%s" % (ctx.attr.native_type),
        "--native_hdr=%s" % (ctx.attr.native_hdr),
        "--degree_over_base_field=%s" % (ctx.attr.degree_over_base_field),
        "--hdr_tpl_path=%s" % (hdr_tpl_path),
        "--src_tpl_path=%s" % (src_tpl_path),
        "--type_traits_hdr_tpl_path=%s" % (type_traits_hdr_tpl_path),
    ]

    ctx.actions.run(
        inputs = [
            ctx.files.hdr_tpl_path[0],
            ctx.files.src_tpl_path[0],
            ctx.files.type_traits_hdr_tpl_path[0],
        ],
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        outputs = [ctx.outputs.out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

generate_ext_field = rule(
    implementation = _generate_ext_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "base_field_class_name": attr.string(mandatory = True),
        "display_name": attr.string(mandatory = True),
        "base_field_display_name": attr.string(mandatory = True),
        "c_base_field_hdr": attr.string(mandatory = True),
        "native_type": attr.string(mandatory = True),
        "native_hdr": attr.string(mandatory = True),
        "degree_over_base_field": attr.int(mandatory = True),
        "hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field.h.tpl"),
        ),
        "src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field.cc.tpl"),
        ),
        "type_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator:ext_field_type_traits.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/ext_field_generator"),
        ),
    },
)

def generate_ext_fields(
        name,
        degree_over_base_field,
        class_name,
        base_field_class_name,
        display_name,
        base_field_display_name,
        c_base_field_hdr,
        c_base_field_deps,
        native_type,
        native_hdr,
        native_deps):
    for n in [
        ("{}_gen_type_traits_hdr".format(name), "{}_type_traits.h".format(name)),
        ("{}_gen_src".format(name), "{}.cc".format(name)),
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_ext_field(
            degree_over_base_field = degree_over_base_field,
            class_name = class_name,
            base_field_class_name = base_field_class_name,
            display_name = display_name,
            base_field_display_name = base_field_display_name,
            c_base_field_hdr = c_base_field_hdr,
            native_type = native_type,
            native_hdr = native_hdr,
            name = n[0],
            out = n[1],
        )

    tachyon_cc_library(
        name = name,
        hdrs = [
            ":{}_gen_type_traits_hdr".format(name),
            ":{}_gen_hdr".format(name),
        ],
        srcs = [":{}_gen_src".format(name)],
        deps = c_base_field_deps + native_deps + [
            "//tachyon/c:export",
            "//tachyon/c/base:type_traits_forward",
        ],
    )

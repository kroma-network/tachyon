load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _generate_prime_field_impl(ctx):
    hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field.h.tpl)", [ctx.attr.hdr_tpl_path])
    src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field.cc.tpl)", [ctx.attr.src_tpl_path])
    type_traits_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field_type_traits.h.tpl)", [ctx.attr.type_traits_hdr_tpl_path])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--class_name=%s" % (ctx.attr.class_name),
        "--display_name=%s" % (ctx.attr.display_name),
        "--native_type=%s" % (ctx.attr.native_type),
        "--native_hdr=%s" % (ctx.attr.native_hdr),
        "--limb_nums=%s" % (ctx.attr.limb_nums),
        "--hdr_tpl_path=%s" % (hdr_tpl_path),
        "--src_tpl_path=%s" % (src_tpl_path),
        "--type_traits_hdr_tpl_path=%s" % (type_traits_hdr_tpl_path),
    ]

    if len(ctx.attr.curve) > 0:
        arguments.append("--curve=%s" % (ctx.attr.curve))

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

generate_prime_field = rule(
    implementation = _generate_prime_field_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "display_name": attr.string(mandatory = True),
        "curve": attr.string(),
        "native_type": attr.string(mandatory = True),
        "native_hdr": attr.string(mandatory = True),
        "limb_nums": attr.int(default = 0),
        "hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field.h.tpl"),
        ),
        "src_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field.cc.tpl"),
        ),
        "type_traits_hdr_tpl_path": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator:prime_field_type_traits.h.tpl"),
        ),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/c/math/finite_fields/generator/prime_field_generator"),
        ),
    },
)

def generate_ec_prime_fields(
        name,
        curve,
        class_name,
        display_name,
        limb_nums,
        native_type,
        native_hdr,
        native_deps):
    for n in [
        ("{}_gen_type_traits_hdr".format(name), "{}_type_traits.h".format(name)),
        ("{}_gen_src".format(name), "{}.cc".format(name)),
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
    ]:
        generate_prime_field(
            curve = curve,
            class_name = class_name,
            display_name = display_name,
            limb_nums = limb_nums,
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
        deps = native_deps + [
            "//tachyon/c:export",
            "//tachyon/c/base:type_traits_forward",
        ],
    )

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

SMALL_SUBGROUP_ADICITY = "small_subgroup_adicity"
SMALL_SUBGROUP_BASE = "small_subgroup_base"
SUBGROUP_GENERATOR = "subgroup_generator"

_PRIME_FIELD = 0
_FFT_PRIME_FIELD = 1
_LARGE_FFT_PRIME_FIELD = 2

def _do_generate_prime_field_impl(ctx, type):
    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--class=%s" % (ctx.attr.class_name),
        "--modulus=%s" % (ctx.attr.modulus),
    ]

    if type >= _FFT_PRIME_FIELD:
        arguments.append("--subgroup_generator=%s" % (ctx.attr.subgroup_generator[BuildSettingInfo].value))

    if type >= _LARGE_FFT_PRIME_FIELD:
        arguments.append("--small_subgroup_adicity=%s" % (ctx.attr.small_subgroup_adicity[BuildSettingInfo].value))
        arguments.append("--small_subgroup_base=%s" % (ctx.attr.small_subgroup_base[BuildSettingInfo].value))

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

def _generate_prime_field_impl(ctx):
    _do_generate_prime_field_impl(ctx, _PRIME_FIELD)

def _generate_fft_prime_field_impl(ctx):
    _do_generate_prime_field_impl(ctx, _FFT_PRIME_FIELD)

def _generate_large_fft_prime_field_impl(ctx):
    _do_generate_prime_field_impl(ctx, _LARGE_FFT_PRIME_FIELD)

def _attrs(type):
    d = {
        "out": attr.output(mandatory = True),
        "namespace": attr.string(mandatory = True),
        "class_name": attr.string(mandatory = True),
        "modulus": attr.string(mandatory = True),
        "hdr_include_override": attr.string(),
        "special_prime_override": attr.string(),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator"),
        ),
    }

    if type >= _FFT_PRIME_FIELD:
        d |= {
            "subgroup_generator": attr.label(),
        }

    if type >= _LARGE_FFT_PRIME_FIELD:
        d |= {
            "small_subgroup_adicity": attr.label(),
            "small_subgroup_base": attr.label(),
        }

    return d

generate_prime_field = rule(
    implementation = _generate_prime_field_impl,
    attrs = _attrs(_PRIME_FIELD),
)

generate_fft_prime_field = rule(
    implementation = _generate_fft_prime_field_impl,
    attrs = _attrs(_FFT_PRIME_FIELD),
)

generate_large_fft_prime_field = rule(
    implementation = _generate_large_fft_prime_field_impl,
    attrs = _attrs(_LARGE_FFT_PRIME_FIELD),
)

def _do_generate_prime_fields(
        name,
        deps = [],
        **kwargs):
    tachyon_cc_library(
        name = name,
        hdrs = [":{}_gen_hdr".format(name)],
        deps = deps + [
            "//tachyon/math/finite_fields:prime_field",
        ],
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

    _do_generate_prime_fields(name, deps, **kwargs)

def generate_fft_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        subgroup_generator,
        hdr_include_override = "",
        special_prime_override = "",
        deps = [],
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_fft_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            subgroup_generator = subgroup_generator,
            hdr_include_override = hdr_include_override,
            special_prime_override = special_prime_override,
            name = n[0],
            out = n[1],
        )

    _do_generate_prime_fields(name, deps, **kwargs)

def generate_large_fft_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        small_subgroup_adicity,
        small_subgroup_base,
        subgroup_generator,
        hdr_include_override = "",
        special_prime_override = "",
        deps = [],
        **kwargs):
    for n in [
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
    ]:
        generate_large_fft_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            small_subgroup_adicity = small_subgroup_adicity,
            small_subgroup_base = small_subgroup_base,
            subgroup_generator = subgroup_generator,
            hdr_include_override = hdr_include_override,
            special_prime_override = special_prime_override,
            name = n[0],
            out = n[1],
        )

    _do_generate_prime_fields(name, deps, **kwargs)

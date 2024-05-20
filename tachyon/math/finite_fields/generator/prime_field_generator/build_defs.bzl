load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@iden3_ffiasm//:build_defs.bzl", generate_ffiasm_prime_field = "generate_prime_field")
load("//bazel:tachyon.bzl", "if_has_asm_prime_field")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

SMALL_SUBGROUP_ADICITY = "small_subgroup_adicity"
SMALL_SUBGROUP_BASE = "small_subgroup_base"
SUBGROUP_GENERATOR = "subgroup_generator"

_PRIME_FIELD = 0
_FFT_PRIME_FIELD = 1
_LARGE_FFT_PRIME_FIELD = 2

def _do_generate_prime_field_impl(ctx, type):
    x86_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_x86.h.tpl)", [ctx.attr.x86_hdr_tpl])
    fail_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:fail.h.tpl)", [ctx.attr.fail_hdr_tpl])
    fail_src_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:fail.cc.tpl)", [ctx.attr.fail_src_tpl])
    config_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:config.h.tpl)", [ctx.attr.config_hdr_tpl])
    cpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_cpu.h.tpl)", [ctx.attr.cpu_hdr_tpl])
    small_cpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:small_prime_field_cpu.h.tpl)", [ctx.attr.small_cpu_hdr_tpl])
    gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_gpu.h.tpl)", [ctx.attr.gpu_hdr_tpl])
    small_gpu_hdr_tpl_path = ctx.expand_location("$(location @kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:small_prime_field_gpu.h.tpl)", [ctx.attr.small_gpu_hdr_tpl])

    arguments = [
        "--out=%s" % (ctx.outputs.out.path),
        "--namespace=%s" % (ctx.attr.namespace),
        "--class=%s" % (ctx.attr.class_name),
        "--modulus=%s" % (ctx.attr.modulus),
        "--flag=%s" % (ctx.attr.flag),
        "--x86_hdr_tpl_path=%s" % (x86_hdr_tpl_path),
        "--fail_hdr_tpl_path=%s" % (fail_hdr_tpl_path),
        "--fail_src_tpl_path=%s" % (fail_src_tpl_path),
        "--config_hdr_tpl_path=%s" % (config_hdr_tpl_path),
        "--cpu_hdr_tpl_path=%s" % (cpu_hdr_tpl_path),
        "--small_cpu_hdr_tpl_path=%s" % (small_cpu_hdr_tpl_path),
        "--gpu_hdr_tpl_path=%s" % (gpu_hdr_tpl_path),
        "--small_gpu_hdr_tpl_path=%s" % (small_gpu_hdr_tpl_path),
    ]

    if ctx.attr.use_asm:
        if ctx.attr._has_asm_prime_field[BuildSettingInfo].value:
            arguments.append("--use_asm")

    if ctx.attr.use_montgomery:
        arguments.append("--use_montgomery")

    if len(ctx.attr.reduce32) > 0:
        arguments.append("--reduce32=%s" % (ctx.attr.reduce32))

    if len(ctx.attr.reduce64) > 0:
        arguments.append("--reduce64=%s" % (ctx.attr.reduce64))

    if type >= _FFT_PRIME_FIELD:
        arguments.append("--subgroup_generator=%s" % (ctx.attr.subgroup_generator[BuildSettingInfo].value))

    if type >= _LARGE_FFT_PRIME_FIELD:
        arguments.append("--small_subgroup_adicity=%s" % (ctx.attr.small_subgroup_adicity[BuildSettingInfo].value))
        arguments.append("--small_subgroup_base=%s" % (ctx.attr.small_subgroup_base[BuildSettingInfo].value))

    ctx.actions.run(
        inputs = [
            ctx.files.x86_hdr_tpl[0],
            ctx.files.fail_hdr_tpl[0],
            ctx.files.fail_src_tpl[0],
            ctx.files.config_hdr_tpl[0],
            ctx.files.cpu_hdr_tpl[0],
            ctx.files.small_cpu_hdr_tpl[0],
            ctx.files.gpu_hdr_tpl[0],
            ctx.files.small_gpu_hdr_tpl[0],
        ],
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
        "flag": attr.string(mandatory = True),
        "reduce32": attr.string(mandatory = True),
        "reduce64": attr.string(mandatory = True),
        "use_asm": attr.bool(mandatory = True),
        "use_montgomery": attr.bool(mandatory = True),
        "x86_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_x86.h.tpl"),
        ),
        "fail_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:fail.h.tpl"),
        ),
        "fail_src_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:fail.cc.tpl"),
        ),
        "config_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:config.h.tpl"),
        ),
        "cpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_cpu.h.tpl"),
        ),
        "small_cpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:small_prime_field_cpu.h.tpl"),
        ),
        "gpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:prime_field_gpu.h.tpl"),
        ),
        "small_gpu_hdr_tpl": attr.label(
            allow_single_file = True,
            default = Label("@kroma_network_tachyon//tachyon/math/finite_fields/generator/prime_field_generator:small_prime_field_gpu.h.tpl"),
        ),
        "_has_asm_prime_field": attr.label(
            default = Label("@kroma_network_tachyon//:has_asm_prime_field"),
        ),
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
        namespace,
        modulus,
        use_asm,
        use_montgomery,
        **kwargs):
    is_small_prime_field = int(modulus) < 1 << 32

    tachyon_cc_library(
        name = "{}_config".format(name),
        hdrs = [":{}_gen_config_hdr".format(name)],
        deps = [
            "//tachyon:export",
            "//tachyon/build:build_config",
            "//tachyon/math/base:big_int",
        ],
    )

    if is_small_prime_field:
        if use_montgomery:
            small_prime_field_dep = "//tachyon/math/finite_fields:small_prime_field_mont"
        else:
            small_prime_field_dep = "//tachyon/math/finite_fields:small_prime_field"

        tachyon_cc_library(
            name = name,
            hdrs = [":{}_gen_hdr".format(name)],
            deps = [
                ":{}_config".format(name),
                small_prime_field_dep,
            ],
            **kwargs
        )
    elif use_asm:
        prefix = namespace.replace("::", "_") + "_" + name
        generate_ffiasm_prime_field(
            name = prefix,
            asm_out = "{}.asm".format(name),
            modulus = modulus,
        )

        cmd_linux_x86 = "\n".join([
            "for out in $(OUTS); do",
            "$(location @nasm//:nasm) -f elf64 -o $$out $$(dirname $(location " + name + ".asm))/$$(basename $${out%.o}.asm)",
            "done",
        ])
        cmd_macos_x86 = "\n".join([
            "for out in $(OUTS); do",
            "$(location @nasm//:nasm) -f macho64 --prefix _ -o $$out $$(dirname $(location " + name + ".asm))/$$(basename $${out%.o}.asm)",
            "done",
        ])

        native.genrule(
            name = "{}_asm".format(name),
            srcs = ["{}.asm".format(name)],
            outs = ["{}.o".format(name)],
            cmd = select({
                "@kroma_network_tachyon//:linux_x86_64": cmd_linux_x86,
                "@kroma_network_tachyon//:macos_x86_64": cmd_macos_x86,
                "//conditions:default": "touch $@",
            }),
            tools = ["@nasm"],
        )

        tachyon_cc_library(
            name = "{}_object".format(name),
            srcs = select({
                "@kroma_network_tachyon//:linux_x86_64": ["{}.o".format(name)],
                "@kroma_network_tachyon//:macos_x86_64": ["{}.o".format(name)],
                "//conditions:default": [],
            }),
            linkstatic = True,
        )

        tachyon_cc_library(
            name = "{}_fail".format(name),
            srcs = select({
                "@kroma_network_tachyon//:linux_x86_64": [":{}_gen_fail_src".format(name)],
                "@kroma_network_tachyon//:macos_x86_64": [":{}_gen_fail_src".format(name)],
                "//conditions:default": [],
            }),
            hdrs = select({
                "@kroma_network_tachyon//:linux_x86_64": [":{}_gen_fail_hdr".format(name)],
                "@kroma_network_tachyon//:macos_x86_64": [":{}_gen_fail_hdr".format(name)],
                "//conditions:default": [],
            }),
            deps = ["//tachyon/base:logging"],
        )

        tachyon_cc_library(
            name = "{}_ffiasm_impl".format(name),
            hdrs = [
                ":{}_gen_hdr".format(name),
            ] + select({
                "@kroma_network_tachyon//:linux_x86_64": [":{}_gen_prime_field_x86_hdr".format(name)],
                "@kroma_network_tachyon//:macos_x86_64": [":{}_gen_prime_field_x86_hdr".format(name)],
                "//conditions:default": [],
            }),
            deps = [
                ":{}_config".format(name),
            ] + select({
                "@kroma_network_tachyon//:linux_x86_64": [
                    ":{}_object".format(name),
                    ":{}_fail".format(name),
                    "//tachyon/math/finite_fields:prime_field_base",
                ],
                "@kroma_network_tachyon//:macos_x86_64": [
                    ":{}_object".format(name),
                    ":{}_fail".format(name),
                    "//tachyon/math/finite_fields:prime_field_base",
                ],
                "//conditions:default": ["//tachyon/math/finite_fields:prime_field_fallback"],
            }),
            **kwargs
        )

        tachyon_cc_library(
            name = "{}_fallback_impl".format(name),
            hdrs = [":{}_gen_hdr".format(name)],
            deps = [
                ":{}_config".format(name),
                "//tachyon/math/finite_fields:prime_field_fallback",
            ],
            **kwargs
        )

        tachyon_cc_library(
            name = name,
            deps = if_has_asm_prime_field(
                [":{}_ffiasm_impl".format(name)],
                [":{}_fallback_impl".format(name)],
            ),
        )
    else:
        tachyon_cc_library(
            name = name,
            hdrs = [":{}_gen_hdr".format(name)],
            deps = [
                ":{}_config".format(name),
                "//tachyon/math/finite_fields:prime_field_fallback",
            ],
            **kwargs
        )

    if is_small_prime_field:
        prime_field_gpu_dep = small_prime_field_dep
    else:
        prime_field_gpu_dep = "//tachyon/math/finite_fields:prime_field_gpu"

    tachyon_cc_library(
        name = "{}_gpu".format(name),
        hdrs = [":{}_gen_gpu_hdr".format(name)],
        deps = [
            ":{}_config".format(name),
            prime_field_gpu_dep,
        ],
        **kwargs
    )

def _gen_name_out_pairs(name):
    return [
        ("{}_gen_config_hdr".format(name), "{}_config.h".format(name)),
        ("{}_gen_hdr".format(name), "{}.h".format(name)),
        ("{}_gen_gpu_hdr".format(name), "{}_gpu.h".format(name)),
        ("{}_gen_prime_field_x86_hdr".format(name), "{}_prime_field_x86.h".format(name)),
        ("{}_gen_fail_hdr".format(name), "{}_fail.h".format(name)),
        ("{}_gen_fail_src".format(name), "{}_fail.cc".format(name)),
    ]

def generate_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        flag,
        reduce32 = "",
        reduce64 = "",
        use_asm = True,
        use_montgomery = True,
        **kwargs):
    for n in _gen_name_out_pairs(name):
        generate_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            flag = flag,
            reduce32 = reduce32,
            reduce64 = reduce64,
            use_asm = use_asm,
            use_montgomery = use_montgomery,
            name = n[0],
            out = n[1],
        )

    _do_generate_prime_fields(name, namespace, modulus, use_asm, use_montgomery, **kwargs)

def generate_fft_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        flag,
        subgroup_generator,
        reduce32 = "",
        reduce64 = "",
        use_asm = True,
        use_montgomery = True,
        **kwargs):
    for n in _gen_name_out_pairs(name):
        generate_fft_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            flag = flag,
            reduce32 = reduce32,
            reduce64 = reduce64,
            use_asm = use_asm,
            use_montgomery = use_montgomery,
            subgroup_generator = subgroup_generator,
            name = n[0],
            out = n[1],
        )

    _do_generate_prime_fields(name, namespace, modulus, use_asm, use_montgomery, **kwargs)

def generate_large_fft_prime_fields(
        name,
        namespace,
        class_name,
        modulus,
        flag,
        small_subgroup_adicity,
        small_subgroup_base,
        subgroup_generator,
        reduce32 = "",
        reduce64 = "",
        use_asm = True,
        use_montgomery = True,
        **kwargs):
    for n in _gen_name_out_pairs(name):
        generate_large_fft_prime_field(
            namespace = namespace,
            class_name = class_name,
            modulus = modulus,
            flag = flag,
            reduce32 = reduce32,
            reduce64 = reduce64,
            use_asm = use_asm,
            use_montgomery = use_montgomery,
            small_subgroup_adicity = small_subgroup_adicity,
            small_subgroup_base = small_subgroup_base,
            subgroup_generator = subgroup_generator,
            name = n[0],
            out = n[1],
        )

    _do_generate_prime_fields(name, namespace, modulus, use_asm, use_montgomery, **kwargs)

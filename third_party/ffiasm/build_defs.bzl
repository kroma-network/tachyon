load("@iden3_ffiasm//:build_defs.bzl", _generate_prime_field = "generate_prime_field")

def generate_prime_field(name, modulus):
    _generate_prime_field(
        name = name.capitalize(),
        arm64_s_out = "{}_raw_arm64.s".format(name),
        asm_out = "{}.asm".format(name),
        c_out = "{}.cpp".format(name),
        element_h_out = "{}_element.hpp".format(name),
        generic_c_out = "{}_generic.cpp".format(name),
        h_out = "{}.hpp".format(name),
        modulus = modulus,
        raw_generic_c_out = "{}_raw_generic.cpp".format(name),
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

    native.cc_library(
        name = "{}_object".format(name),
        srcs = select({
            "@kroma_network_tachyon//:linux_x86_64": ["{}.o".format(name)],
            "@kroma_network_tachyon//:macos_x86_64": ["{}.o".format(name)],
            "//conditions:default": [],
        }),
        linkstatic = True,
    )

    native.cc_library(
        name = name,
        srcs = ["{}.cpp".format(name)] + select({
            "@kroma_network_tachyon//:linux_x86_64": [],
            "@kroma_network_tachyon//:macos_x86_64": [],
            "//conditions:default": [
                "{}_generic.cpp".format(name),
                "{}_raw_generic.cpp".format(name),
            ],
        }),
        hdrs = [
            "{}.hpp".format(name),
            "{}_element.hpp".format(name),
        ],
        defines = select({
            "@kroma_network_tachyon//:linux_x86_64": ["USE_ASM", "ARCH_X86_64"],
            "@kroma_network_tachyon//:macos_x86_64": ["USE_ASM", "ARCH_X86_64"],
            "//conditions:default": [],
        }),
        includes = ["."],
        deps = ["@local_config_gmp//:gmp"] + select({
            "@kroma_network_tachyon//:linux_x86_64": [":{}_object".format(name)],
            "@kroma_network_tachyon//:macos_x86_64": [":{}_object".format(name)],
            "//conditions:default": [],
        }),
    )

load("@iden3_ffiasm//:build_defs.bzl", "generate_prime_field")

def fr_library(modulus):
    generate_prime_field(
        name = "Fr",
        asm_out = "fr.asm",
        h_out = "fr.hpp",
        c_out = "fr.cpp",
        element_h_out = "fr_element.hpp",
        generic_c_out = "fr_generic.cpp",
        raw_generic_c_out = "fr_raw_generic.cpp",
        arm64_s_out = "fr_raw_arm64.s",
        modulus = modulus,
    )

    cmd_linux_x86 = "\n".join([
        "for out in $(OUTS); do",
        "$(location @nasm//:nasm) -f elf64 -o $$out $$(dirname $(location fr.asm))/$$(basename $${out%.o}.asm)",
        "done",
    ])

    cmd_macos_x86 = "\n".join([
        "for out in $(OUTS); do",
        "$(location @nasm//:nasm) -f macho64 --prefix _ -o $$out $$(dirname $(location fr.asm))/$$(basename $${out%.o}.asm)",
        "done",
    ])

    native.genrule(
        name = "fr_asm",
        srcs = ["fr.asm"],
        outs = ["fr.o"],
        cmd = select({
            "@kroma_network_tachyon//:linux_x86_64": cmd_linux_x86,
            "@kroma_network_tachyon//:macos_x86_64": cmd_macos_x86,
            "//conditions:default": "touch $@",
        }),
        tools = ["@nasm"],
    )

    native.cc_library(
        name = "fr_object",
        srcs = select({
            "@kroma_network_tachyon//:linux_x86_64": ["fr.o"],
            "@kroma_network_tachyon//:macos_x86_64": ["fr.o"],
            "//conditions:default": [],
        }),
        linkstatic = True,
    )

    native.cc_library(
        name = "fr",
        srcs = ["fr.cpp"] + select({
            "@kroma_network_tachyon//:linux_x86_64": [],
            "@kroma_network_tachyon//:macos_x86_64": [],
            "//conditions:default": [
                "fr_generic.cpp",
                "fr_raw_generic.cpp",
            ],
        }),
        defines = select({
            "@kroma_network_tachyon//:linux_x86_64": ["USE_ASM", "ARCH_X86_64"],
            "@kroma_network_tachyon//:macos_x86_64": ["USE_ASM", "ARCH_X86_64"],
            "//conditions:default": [],
        }),
        hdrs = [
            "fr.hpp",
            "fr_element.hpp",
        ],
        includes = ["."],
        deps = ["@local_config_gmp//:gmp"] + select({
            "@kroma_network_tachyon//:linux_x86_64": [":fr_object"],
            "@kroma_network_tachyon//:macos_x86_64": [":fr_object"],
            "//conditions:default": [],
        }),
    )

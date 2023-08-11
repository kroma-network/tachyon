load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "assemble",
    srcs = [
        "src/ffiasm/fec.asm",
        "src/ffiasm/fnec.asm",
        "src/ffiasm/fq.asm",
        "src/ffiasm/fr.asm",
    ],
    outs = [
        "src/ffiasm/fec.o",
        "src/ffiasm/fnec.o",
        "src/ffiasm/fq.o",
        "src/ffiasm/fr.o",
    ],
    cmd = "for out in $(OUTS); do\n" +
          "  $(location @nasm//:nasm) -f elf64" +
          "    -o $$out" +
          "    $$(dirname $(location src/ffiasm/fq.asm))/$$(basename $${out%.o}.asm)\n" +
          "done",
    tools = ["@nasm"],
)

cc_library(
    name = "fec",
    srcs = ["src/ffiasm/fec.o"],
    linkstatic = True,
)

cc_library(
    name = "fnec",
    srcs = ["src/ffiasm/fnec.o"],
    linkstatic = True,
)

cc_library(
    name = "fq",
    srcs = ["src/ffiasm/fq.o"],
    linkstatic = True,
)

cc_library(
    name = "fr",
    srcs = ["src/ffiasm/fr.o"],
    linkstatic = True,
)

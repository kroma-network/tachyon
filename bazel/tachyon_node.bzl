load("//bazel:tachyon.bzl", "if_node_binding")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library", "tachyon_cc_shared_library")

def tachyon_node_library(
        name,
        testonly = False,
        visibility = ["//visibility:public"],
        **kwrargs):
    tachyon_cc_library(
        name = name + "_lib",
        testonly = testonly,
        **kwrargs
    )

    tachyon_cc_shared_library(
        name = name + "_shared_lib",
        deps = [":{}_lib".format(name)] + if_node_binding([
            "@node_addon_api",
        ]),
        testonly = testonly,
    )

    native.genrule(
        name = name,
        srcs = [":{}_shared_lib".format(name)],
        outs = [name + ".node"],
        cmd = "cp -f $< $@",
        testonly = testonly,
        visibility = visibility,
    )

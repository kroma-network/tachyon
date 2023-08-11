load("//tachyon/build:buildflag.bzl", "attrs", "gen_buildflag_header_helper", "get_var")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")

def _gen_time_buildflag_header_impl(ctx):
    names = ["ENABLE_MACH_ABSOLUTE_TIME_TICKS"]
    return gen_buildflag_header_helper(ctx, ["%s=%s" % (name, get_var(ctx, name.lower())) for name in names])

_gen_time_buildflag_header = rule(
    implementation = _gen_time_buildflag_header_impl,
    output_to_genfiles = True,
    attrs = attrs(),
)

def time_buildflag_header(name):
    _gen_time_buildflag_header(
        name = "gen_" + name,
        out = name + ".h",
    )

    tachyon_cc_library(
        name = name,
        hdrs = [name + ".h"],
        visibility = ["//visibility:public"],
        deps = ["//tachyon/build:buildflag"],
    )

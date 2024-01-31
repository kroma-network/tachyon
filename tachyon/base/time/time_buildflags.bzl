load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//bazel:tachyon_cc.bzl", "tachyon_cc_library")
load("//tachyon/build:buildflag.bzl", "attrs", "gen_buildflag_header_helper")

ENABLE_MACH_ABSOLUTE_TIME_TICKS = "enable_mach_absolute_time_ticks"

def _gen_time_buildflag_header_impl(ctx):
    names = [ENABLE_MACH_ABSOLUTE_TIME_TICKS]
    values = [ctx.attr.enable_mach_absolute_time_ticks]

    return gen_buildflag_header_helper(ctx, [
        "%s=%s" % (pair[0].upper(), pair[1][BuildSettingInfo].value)
        for pair in zip(names, values)
    ])

_gen_time_buildflag_header = rule(
    implementation = _gen_time_buildflag_header_impl,
    output_to_genfiles = True,
    attrs = attrs() | {
        ENABLE_MACH_ABSOLUTE_TIME_TICKS: attr.label(),
    },
)

def time_buildflag_header(name, enable_mach_absolute_time_ticks):
    _gen_time_buildflag_header(
        enable_mach_absolute_time_ticks = enable_mach_absolute_time_ticks,
        name = "gen_" + name,
        out = name + ".h",
    )

    tachyon_cc_library(
        name = name,
        hdrs = [name + ".h"],
        visibility = ["//visibility:public"],
        deps = ["//tachyon/build:buildflag"],
    )

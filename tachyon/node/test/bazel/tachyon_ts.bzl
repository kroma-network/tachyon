load("@aspect_rules_swc//swc:defs.bzl", "swc")
load("@aspect_rules_ts//ts:defs.bzl", "ts_project")
load("@bazel_skylib//lib:partial.bzl", "partial")

def tachyon_ts_project(
        name,
        declaration = True,
        source_map = True,
        transpiler = partial.make(
            swc,
            swcrc = "//:.swcrc",
        ),
        tsconfig = "//:tsconfig",
        deps = [],
        **kwargs):
    ts_project(
        name = name,
        declaration = declaration,
        source_map = source_map,
        transpiler = transpiler,
        tsconfig = tsconfig,
        deps = [
            "//:node_modules/@jest/globals",
            "//:node_modules/@types/jest",
            "//:node_modules/@types/node",
        ] + deps,
        **kwargs,
    )

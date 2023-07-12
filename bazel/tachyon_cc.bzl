load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load(
    "//bazel:tachyon.bzl",
    "if_has_exception",
    "if_has_rtti",
    "if_static",
)

def tachyon_safe_code():
    return ["-Wall", "-Werror"]

def tachyon_warnings(safe_code):
    warnings = []
    if safe_code:
        warnings.extend(tachyon_safe_code())
    return warnings

def tachyon_hide_symbols():
    return if_static([], ["-fvisibility=hidden"])

def tachyon_exceptions(force_exceptions):
    return if_has_exception(["-fexceptions"], (["-fexceptions"] if force_exceptions else ["-fno-exceptions"]))

def tachyon_rtti(force_rtti):
    return if_has_rtti(["-frtti"], (["-frtti"] if force_rtti else ["-fno-rtti"]))

def tachyon_copts(safe_code = True):
    return tachyon_warnings(safe_code) + tachyon_hide_symbols()

def tachyon_cxxopts(safe_code = True, force_exceptions = False, force_rtti = False):
    return tachyon_copts(safe_code) + tachyon_exceptions(force_exceptions) + tachyon_rtti(force_rtti)

def tachyon_cuda_defines():
    return ["TACHYON_CUDA"]

def tachyon_defines(use_cuda = False):
    defines = tachyon_defines_component_build()
    if use_cuda:
        defines += if_cuda(tachyon_cuda_defines())
    return defines

def tachyon_defines_component_build():
    return if_static([], ["TACHYON_COMPONENT_BUILD"])

def tachyon_local_defines():
    return tachyon_local_defines_compile_library()

def tachyon_local_defines_compile_library():
    return if_static([], [
        "TACHYON_COMPILE_LIBRARY",
    ])

def tachyon_cc_library(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        use_cuda = False,
        **kwargs):
    native.cc_library(
        name = name,
        copts = copts + tachyon_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + tachyon_defines(use_cuda),
        local_defines = local_defines + tachyon_local_defines(),
        **kwargs
    )

def tachyon_cc_binary(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        use_cuda = False,
        **kwargs):
    native.cc_binary(
        name = name,
        copts = copts + tachyon_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + tachyon_defines(use_cuda),
        local_defines = local_defines + tachyon_local_defines(),
        **kwargs
    )

def tachyon_cc_test(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkstatic = 1,
        deps = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        use_cuda = False,
        **kwargs):
    native.cc_test(
        name = name,
        copts = copts + tachyon_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + tachyon_defines(use_cuda),
        local_defines = local_defines + tachyon_local_defines(),
        linkstatic = linkstatic,
        deps = deps + ["@com_google_googletest//:gtest_main"],
        **kwargs
    )

def tachyon_cuda_library(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        **kwargs):
    cuda_library(
        name = name,
        copts = copts + tachyon_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + tachyon_defines(use_cuda = True),
        local_defines = local_defines + tachyon_local_defines(),
        **kwargs
    )

def tachyon_cuda_binary(
        name,
        srcs = [],
        deps = [],
        **kwargs):
    tachyon_cc_binary(
        name = name,
        srcs = if_cuda(srcs),
        deps = deps + if_cuda([
            "@local_config_cuda//cuda:cudart",
        ]),
        use_cuda = True,
        **kwargs
    )

def tachyon_cuda_test(
        name,
        srcs = [],
        deps = [],
        **kwargs):
    tachyon_cc_test(
        name = name,
        srcs = if_cuda(srcs),
        deps = deps + if_cuda([
            "@local_config_cuda//cuda:cudart",
        ]),
        use_cuda = True,
        **kwargs
    )

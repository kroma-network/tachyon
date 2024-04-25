load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

# See https://semver.org/
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_PRERELEASE = ""
VERSION = ".".join([str(VERSION_MAJOR), str(VERSION_MINOR), str(VERSION_PATCH)])

def if_x86_32(a, b = []):
    return select({
        "@platforms//cpu:x86_32": a,
        "//conditions:default": b,
    })

def if_x86_64(a, b = []):
    return select({
        "@platforms//cpu:x86_64": a,
        "//conditions:default": b,
    })

def if_arm(a, b = []):
    return select({
        "@platforms//cpu:arm": a,
        "//conditions:default": b,
    })

def if_aarch64(a, b = []):
    return select({
        "@platforms//cpu:aarch64": a,
        "//conditions:default": b,
    })

def if_linux(a, b = []):
    return select({
        "@platforms//os:linux": a,
        "//conditions:default": b,
    })

def if_linux_x86_64(a, b = []):
    return select({
        "@kroma_network_tachyon//:linux_x86_64": a,
        "//conditions:default": b,
    })

def if_macos(a, b = []):
    return select({
        "@platforms//os:macos": a,
        "//conditions:default": b,
    })

def if_windows(a, b = []):
    return select({
        "@platforms//os:windows": a,
        "//conditions:default": b,
    })

def if_posix(a, b = []):
    return select({
        "@platforms//os:windows": b,
        "//conditions:default": a,
    })

def if_optimized(a, b = []):
    return select({
        "@kroma_network_tachyon//:optimized": a,
        "//conditions:default": b,
    })

def if_static(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_shared_object": b,
        "@kroma_network_tachyon//:tachyon_c_shared_object": b,
        "@kroma_network_tachyon//:tachyon_cc_shared_object": b,
        "//conditions:default": a,
    })

def if_has_exception(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_exception": a,
        "//conditions:default": b,
    })

def if_has_rtti(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_rtti": a,
        "//conditions:default": b,
    })

def if_has_openmp(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_openmp": a,
        "//conditions:default": b,
    })

def if_has_avx512(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_avx512": a,
        "//conditions:default": b,
    })

def if_has_matplotlib(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_matplotlib": a,
        "//conditions:default": b,
    })

def if_has_numa(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_has_numa": a,
        "//conditions:default": b,
    })

def if_c_shared_object(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_c_shared_object": a,
        "//conditions:default": b,
    })

def if_cc_shared_object(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_cc_shared_object": a,
        "//conditions:default": b,
    })

def if_py_binding(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_py_binding": a,
        "//conditions:default": b,
    })

def if_node_binding(a, b = []):
    return select({
        "@kroma_network_tachyon//:tachyon_node_binding": a,
        "//conditions:default": b,
    })

def if_gpu_is_configured(x):
    return if_cuda_is_configured(x) + if_rocm_is_configured(x)

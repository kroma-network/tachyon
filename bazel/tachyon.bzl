# See https://semver.org/
VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_PATCH = 1
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
        "@com_github_lightscale_tachyon//:optimized": a,
        "//conditions:default": b,
    })

def if_static(a, b = []):
    return select({
        "@com_github_lightscale_tachyon//:tachyon_framework_shared_object": b,
        "//conditions:default": a,
    })

def if_has_exception(a, b = []):
    return select({
        "@com_github_lightscale_tachyon//:tachyon_has_exception": a,
        "//conditions:default": b,
    })

def if_gmp_backend(a, b = []):
    return select({
        "@com_github_lightscale_tachyon//:tachyon_gmp_backend": a,
        "//conditions:default": b,
    })

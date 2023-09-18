load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "node_addon_api",
    hdrs = [
        "node_modules/node-addon-api/napi.h",
        "node_modules/node-addon-api/napi-inl.deprecated.h",
        "node_modules/node-addon-api/napi-inl.h",
    ],
    defines = [
        "TACHYON_NODE_BINDING",
    ] + [
        "NODE_GYP_MODULE_NAME",
        "USING_UV_SHARED=1",
        "USING_V8_SHARED=1",
        "V8_DEPRECATION_WARNINGS=1",
        "V8_DEPRECATION_WARNINGS",
        "V8_IMMINENT_DEPRECATION_WARNINGS",
        "_GLIBCXX_USE_CXX11_ABI=1",
        "_LARGEFILE_SOURCE",
        "_FILE_OFFSET_BITS=64",
        "__STDC_FORMAT_MACROS",
        "OPENSSL_NO_PINSHARED",
        "OPENSSL_THREADS",
        "NAPI_DISABLE_CPP_EXCEPTIONS",
        "BUILDING_NODE_EXTENSION",
    ],
    includes = ["node_modules/node-addon-api"],
    strip_include_prefix = "node_modules/node-addon-api",
    include_prefix = "third_party/node_addon_api",
    visibility = ["//visibility:public"],
    deps = [":node"],
)

cc_library(
    name = "node",
    hdrs = glob(["%{NODE_VERSION}/include/**/*.h"]),
    includes = [
        "%{NODE_VERSION}/include",
        "%{NODE_VERSION}/include/node",
    ],
)

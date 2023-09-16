load("@aspect_rules_jest//jest:defs.bzl", "jest_test")

def tachyon_jest_test(
        name,
        config = "//:jest.config",
        node_modules = "//:node_modules",
        **kwargs):
    jest_test(
        name = name,
        config = config,
        node_modules = node_modules,
        **kwargs
    )

workspace(name = "kroma_network_tachyon")

load("//bazel:tachyon_deps.bzl", "tachyon_deps")

tachyon_deps()

# Start of buildifier
load("//bazel:buildifier_deps.bzl", "buildifier_deps")

buildifier_deps()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.20.3")

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()
# End of buildifier

# Start of rules_rust
load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

# We need to change the default value of flag //tachyon/rs/base:rustc_version_ge_1.67.0
# if we change the default rustc version.
# See //tachyon/rs/base/BUILD.bazel.
rust_register_toolchains(
    edition = "2021",
    versions = [
        "1.66.1",
    ],
)

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")

crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:Cargo.Bazel.lock",
    manifests = [
        "//:Cargo.toml",
        "//benchmark/msm/arkworks:Cargo.toml",
        "//benchmark/msm/bellman:Cargo.toml",
        "//benchmark/msm/halo2:Cargo.toml",
        "//benchmark/fft/arkworks:Cargo.toml",
        "//benchmark/fft/bellman:Cargo.toml",
        "//benchmark/fft/halo2:Cargo.toml",
        "//benchmark/poseidon/arkworks:Cargo.toml",
        "//tachyon/rs:Cargo.toml",
        "//vendors/halo2:Cargo.toml",
    ],
    packages = {
        # See https://github.com/bazelbuild/rules_rust/issues/2071#issuecomment-1656204269
        "serde": crate.spec(version = "=1.0.164"),
    },
)

load(
    "@crate_index//:defs.bzl",
    tachyon_crate_repositories = "crate_repositories",
)

tachyon_crate_repositories()
# End of rules_rust

load("@cxx.rs//third-party/bazel:defs.bzl", cxx_create_repositories = "crate_repositories")

cxx_create_repositories()

load("//bazel:pybind11_deps.bzl", "pybind11_deps")

pybind11_deps()

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

# Start of rules_js
load("//bazel:js_deps.bzl", "js_deps")

js_deps()

load("@aspect_rules_js//js:repositories.bzl", "rules_js_dependencies")

rules_js_dependencies()

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies")

aspect_bazel_lib_dependencies()

# Fetch and register node, if you haven't already
load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = DEFAULT_NODE_VERSION,
)

load("@aspect_rules_js//npm:repositories.bzl", "npm_translate_lock")

npm_translate_lock(
    name = "npm",
    data = ["@iden3_ffiasm//:package.json"],
    npm_package_lock = "@iden3_ffiasm//:package-lock.json",
    pnpm_lock = "@iden3_ffiasm//:pnpm-lock.yaml",
    update_pnpm_lock = True,
    verify_node_modules_ignored = "@iden3_ffiasm//:.bazelignore",
)

load("@npm//:repositories.bzl", "npm_repositories")

npm_repositories()

# End of rules_js

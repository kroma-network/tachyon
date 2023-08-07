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

rust_register_toolchains(
    edition = "2021",
    versions = [
        "1.66.1",
    ],
)

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crates_repository")

crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:Cargo.Bazel.lock.json",
    manifests = [
        "//:Cargo.toml",
        "//:benchmark/msm/arkworks/Cargo.toml",
        "//:vendors/halo2/Cargo.toml",
    ],
)

load(
    "@crate_index//:defs.bzl",
    tachyon_crate_repositories = "crate_repositories",
)

tachyon_crate_repositories()
# End of rules_rust

load("@cxx.rs//third-party/bazel:defs.bzl", cxx_crate_repositories = "crate_repositories")

cxx_crate_repositories()

load("//bazel:pybind11_deps.bzl", "pybind11_deps")

pybind11_deps()

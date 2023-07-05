workspace(name = "com_github_lightscale_tachyon")

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

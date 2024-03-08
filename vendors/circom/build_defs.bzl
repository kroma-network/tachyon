def _cpp_dir(name):
    return name + "_cpp/"

def _c_srcs(name):
    return [_cpp_dir(name) + file for file in [
        name + ".cpp",
    ]]

def _c_data(name):
    return [_cpp_dir(name) + file for file in [
        name + ".dat",
    ]]

def _compile_circuit_impl(ctx):
    arguments = ["--r1cs", "--c"]

    if ctx.attr.O0:
        arguments.append("--O0")
    if ctx.attr.O1:
        arguments.append("--O1")
    if ctx.attr.O2:
        arguments.append("--O2")
    if ctx.attr.verbose:
        arguments.append("--verbose")
    if ctx.attr.inspect:
        arguments.append("--inspect")
    if ctx.attr.use_old_simplification_heuristics:
        arguments.append("--use_old_simplification_heuristics")
    if ctx.attr.simplification_substitution:
        arguments.append("--simplification_substitution")
    if ctx.attr.prime:
        arguments.append("--prime={}".format(ctx.attr.prime))
    if ctx.attr.O2round != 0:
        arguments.append("--O2round={}".format(ctx.attr.O2round))

    src = ctx.attr.main.files.to_list()[0]
    arguments.append(src.path)

    name = src.basename[:-len(".circom")]
    outputs = [
        name + ".r1cs",
    ] + _c_srcs(name) + _c_data(name)
    outputs = [ctx.actions.declare_file(out) for out in outputs]
    arguments.append("--output=" + outputs[0].dirname)

    ctx.actions.run(
        inputs = ctx.files.srcs + ctx.files.main,
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        progress_message = "Compiling for " + name,
        outputs = outputs,
        arguments = arguments,
    )

    return [DefaultInfo(files = depset(outputs))]

compile_circuit = rule(
    implementation = _compile_circuit_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "main": attr.label(mandatory = True, allow_single_file = True),
        "O0": attr.bool(doc = "No simplification is applied"),
        "O1": attr.bool(doc = "Only applies signal to signal and signal to constant simplification"),
        "O2": attr.bool(doc = "Full constraint simplification"),
        "verbose": attr.bool(doc = "Shows logs during compilation"),
        "inspect": attr.bool(doc = "Does an additional check over the constraints produced"),
        "use_old_simplification_heuristics": attr.bool(
            doc = "Applies the old version of the heuristics when performing linear simplification",
        ),
        "simplification_substitution": attr.bool(
            doc = "Outputs the substitution applied in the simplification phase in json",
        ),
        "prime": attr.string(
            doc = "To choose the prime number to use to generate the circuit.",
            values = ["bn128", "bls12381", "goldilocks", "grumpkin", "pallas", "vesta", "secq256r1"],
            default = "bn128",
        ),
        "O2round": attr.int(doc = "Maximum number of rounds of the simplification process"),
        "_tool": attr.label(
            # TODO(chokobole): Change to "exec", so we can build on macos.
            cfg = "target",
            executable = True,
            allow_single_file = True,
            default = Label("@kroma_network_circom//:circom"),
        ),
    },
)

def witness_gen_library(
        name,
        gendep,
        prime = "bn128"):
    native.cc_library(
        name = name,
        srcs = [
            gendep,
            "//circomlib/generated/common:common_srcs",
        ],
        data = [gendep],
        deps = [
            "//circomlib/generated/common:common_hdrs",
            "//circomlib/generated/{}:fr".format(prime),
            "@com_google_absl//absl/container:flat_hash_map",
        ],
    )

def _generate_r1cs_impl(ctx):
    arguments = [
        "--r1cs",
    ]

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

    dst = src.basename[:-len(".circom")] + ".r1cs"
    out = ctx.actions.declare_file(dst)

    arguments.append("--output=" + out.dirname)

    ctx.actions.run(
        inputs = ctx.files.srcs,
        tools = [ctx.executable._tool],
        executable = ctx.executable._tool,
        progress_message = "Generating %s" % (out.short_path),
        outputs = [out],
        arguments = arguments,
    )

    return [DefaultInfo(files = depset([out]))]

generate_r1cs = rule(
    implementation = _generate_r1cs_impl,
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

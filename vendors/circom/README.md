# Circom

## Features

|                           | Tachyon   | Circom-compat | Rapidsnark |
| ------------------------- | --------- | ------------- | ---------- |
| Language                  | C++       | Rust          | C++        |
| Witness Generator         | C(Fast)   | WASM(Slow)    | -          |
| Flexible Witness Handling | Yes       | Yes           | No         |
| Field Support             | All       | Bn128         | Bn128      |
| [FFIASM] Generated Field  | Yes(Fast) | No(Slow)      | Yes(Fast)  |

[FFIASM]: https://github.com/iden3/ffiasm

1. Speed: Tachyon runs approximately 10 times faster than Rapidsnark! I conducted a [benchmark](/vendors/circom/benchmark/README.md) using sha256_512.circom with a degree of 2¹⁶.
2. Seamless Workflow: Tachyon doesn't require you to run any additional programs just to generate a circom proof. No need for snarkjs for witness file generation.
3. Flexible Witness Handling: With Tachyon, you can modify the witness directly in your program. No need to create a separate JSON file with snarkjs.
4. Integrated Build System: Tachyon seamlessly integrates circom compilation into the build system, specifically bazel. When you make changes to a circom file, it's compiled accordingly and built in parallel, ensuring a safe and efficient process.
5. Field Support: Tachyon isn't limited to bn128; it easily supports all fields!

## How to migrate

### Migration Steps

1. Clone [circom-examlpes](https://github.com/kroma-network/circom-example) as a template.

   ```shell
   git clone https://github.com/kroma-network/circom-example
   cd circom-example
   git submodule update --init
   ```

2. Prepare your circuit. For this example, let's use [adder.circom](/vendors/circom/examples/adder.circom).

   ```shell
   cp /path/to/adder.circom /path/to/circom-example/circuits/adder.circom
   ```

3. Follow the [instruction](https://docs.circom.io/getting-started/proving-circuits/) to generate a zkey.
   In this example, we'll use an existing [adder.zkey](/vendors/circom/examples/adder.zkey).

   ```shell
   cp /path/to/adder.zkey /path/to/circom-example/circuits/adder.zkey
   ```

4. Update [rules_circom/circuits/BUILD.bazel](https://github.com/kroma-network/circom-example/blob/main/circuits/BUILD.bazel) as follows:

   ```diff
   -exports_files(["multiplier_2.zkey"])
   +exports_files([
   +    "adder.zkey",
   +    "multiplier_2.zkey",
   +])

   PRIME = "bn128"

   +compile_circuit(
   +    name = "compile_adder",
   +    main = "adder.circom",
   +    prime = PRIME,
   +)

   +witness_gen_library(
   +    name = "gen_witness_adder",
   +    gendep = ":compile_adder",
   +    prime = PRIME,
   +)
   ```

   See [rules_circom](https://github.com/kroma-network/rules_circom) for more details about `compile_circuit`.

5. Update [circom-example/src/prover_main.cc](https://github.com/kroma-network/circom-example/blob/main/src/prover_main.cc) as follows:

   ```diff
   WitnessLoader<F> witness_loader(
   -    base::FilePath("circuits/multiplier_2_cpp/multiplier_2.dat"));
   +    base::FilePath("circuits/adder_cpp/adder.dat"));
   witness_loader.Set("in1", a);
   witness_loader.Set("in2", b);
   witness_loader.Load();

   zk::r1cs::groth16::ProvingKey<Curve> proving_key;
   zk::r1cs::ConstraintMatrices<F> constraint_matrices;
   {
     ZKeyParser zkey_parser;
     std::unique_ptr<ZKey> zkey =
   -      zkey_parser.Parse(base::FilePath("circuits/multiplier_2.zkey"));
   +      zkey_parser.Parse(base::FilePath("circuits/adder.zkey"));
   ```

6. Update [rules_circom/src/BUILD.bazel](https://github.com/kroma-network/circom-example/blob/main/src/BUILD.bazel) as follows:

   ```diff
   tachyon_cc_binary(
       name = "prover_main",
       srcs = ["prover_main.cc"],
       data = [
   -        "//circuits:compile_multiplier_2",
   -        "//circuits:multiplier_2.zkey",
   +        "//circuits:compile_adder",
   +        "//circuits:adder.zkey",
       ],
       deps = [
   -        "//circuits:gen_witness_multiplier_2",
   +        "//circuits:gen_witness_adder",
           "@kroma_network_circom//circomlib/circuit:quadratic_arithmetic_program",
           "@kroma_network_circom//circomlib/circuit:witness_loader",
   ```

7. Run this command for the first time only.

   ```shell
   CARGO_BAZEL_REPIN=1 bazel sync --only=crate_index
   ```

8. Finally, generate the proof.

   ```shell
   bazel run //src:prover_main
   ```

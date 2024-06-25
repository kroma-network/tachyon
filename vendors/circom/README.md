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

1. Speed: Tachyon runs approximately 10 times faster than Rapidsnark! The [benchmark](/vendors/circom/benchmark/README.md) was conducted using sha256_512.circom with a degree of 2¹⁶.
2. Seamless Workflow: Tachyon doesn't require you to run any additional programs just to generate a circom proof. No need for snarkjs for witness file generation.
3. Flexible Witness Handling: With Tachyon, you can modify the witness directly in your program. No need to create a separate JSON file with snarkjs.
4. Integrated Build System: Tachyon seamlessly integrates circom compilation into the build system, specifically bazel. When you make changes to a circom file, it's compiled accordingly and built in parallel, ensuring a safe and efficient process.
5. Field Support: Tachyon isn't limited to bn128; it easily supports all fields!

## How to build

Go to [prerequisites](../../docs/how_to_use/how_to_build.md#Prerequisites) and follow the instructions.

### Linux

```shell
bazel build --@kroma_network_tachyon//:has_openmp -c opt --config linux //:prover_main
```

### MacOS arm64

```shell
bazel build -c opt --config macos_arm64 //:prover_main
```

### MacOS x64

```shell
bazel build -c opt --config macos_x86_64 //:prover_main
```

## How to run

```shell
bazel-bin/prover_main -h
Usage:

bazel-bin/prover_main zkey wtns proof public -n [OPTIONS]

Positional arguments:

zkey                The path to zkey file
wtns                The path to wtns file
proof               The path to proof json
public              The path to public json
n                   The number of times to run the proof generation


Optional arguments:

--curve             The curve type among ('bn254', bls12_381'), by default 'bn254'
--no_zk             Create proof without zk. By default zk is enabled. Use this flag in case you want to compare the proof with rapidsnark.
--verify            Verify the proof. By default verify is disabled. Use this flag to verify the proof with the public inputs.
```

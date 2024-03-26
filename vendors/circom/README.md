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

## How to migrate

See [circom-example](https://github.com/kroma-network/circom-example/) for more details.

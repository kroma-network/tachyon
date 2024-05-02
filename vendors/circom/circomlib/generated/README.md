# Generated

This is taken and modified from [iden3/circom/code_producers/src/c_elements](https://github.com/iden3/circom/tree/v2.1.8/code_producers/src/c_elements).

## Modifications

- Modified to remove errors from compilation.
- `loadCircuit` and `writeBinWitness` are moved to [common/calcwit.hpp](/vendors/circom//circomlib/generated/common/calcwit.hpp) and [common/calcwit.hpp](/vendors/circom//circomlib/generated/common/calcwit.cpp).
- `loadWitness` is created, which loads the witness from the `absl::flat_hash_map<>`.

See the following files for more details on the modifications.

- [calcwit.cpp.diff](/vendors/circom/circomlib/generated/common/calcwit.cpp.diff)
- [calcwit.hpp.diff](/vendors/circom/circomlib/generated/common/calcwit.hpp.diff)
- [circom.hpp.diff](/vendors/circom/circomlib/generated/common/circom.hpp.diff)

# Tracking Testing Types Across Example Circuits

## Currently Tested Types

| Circuit             | FloorPlanner       | PCS     | Vendor | LookupType         |
| ------------------- | ------------------ | ------- | ------ | ------------------ |
| SimpleCircuit       | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| SimpleCircuit       | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| SimpleLookupCircuit | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| SimpleLookupCircuit | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| ShuffleCircuit      | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| ShuffleCircuit      | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| ShuffleAPICircuit   | SimpleFloorPlanner | SHPlonk | Scroll | LogDerivativeHalo2 |
| ShuffleAPICircuit   | V1FloorPlanner     | SHPlonk | Scroll | LogDerivativeHalo2 |
| Fibonacci1Circuit   | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| Fibonacci1Circuit   | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| Fibonacci2Circuit   | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| Fibonacci2Circuit   | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| Fibonacci3Circuit   | SimpleFloorPlanner | SHPlonk | Scroll | Halo2              |
| Fibonacci3Circuit   | V1FloorPlanner     | SHPlonk | Scroll | Halo2              |
| MultiLookupCircuit  | SimpleFloorPlanner | SHPlonk | Scroll | LogDerivativeHalo2 |
| MultiLookupCircuit  | SimpleFloorPlanner | GWC     | Scroll | LogDerivativeHalo2 |

**In Progress**:

| Circuit        | FloorPlanner       | PCS | Vendor | LookupType |
| -------------- | ------------------ | --- | ------ | ---------- |
| ShuffleCircuit | SimpleFloorPlanner | GWC | Scroll | Halo2      |
| ShuffleCircuit | V1FloorPlanner     | GWC | Scroll | Halo2      |

## Identical Test Data

The following circuits result in the same test data when using `SimpleFloorPlanner` or `V1FloorPlanner`

- SimpleLookupCircuit
- ShuffleCircuit
- ShuffleAPICircuit
- Fibonacci2Circuit
- Fibonacci3Circuit

## Test Data Variables

:red_square: Required variables

:blue_square: Optional variables with flags

:green_square: Optional variables without flags (defaulted)

Refer to the following files to learn more about these variables:

- [circuit_test_data.h](circuit_test_data.h)
- [circuit_test.cc](circuit_test.cc)
- Any `...test_data.h` file:
  - [shuffle_circuit_test_data.h](shuffle_circuit_test_data.h)
  - [shuffle_api_circuit_test_data.h](shuffle_api_circuit_test_data.h)
  - [simple_circuit_test_data.h](simple_circuit_test_data.h)
  - [simple_lookup_circuit_test_data.h](simple_lookup_circuit_test_data.h)

1. :red_square: kN
1. :red_square: kPinnedConstraintSystem
1. :blue_square: kAssemblyFixedColumns
1. :blue_square: kAssemblyPermutationColumns
1. :blue_square: kCycleStoreMapping
1. :blue_square: kCycleStoreAux
1. :blue_square: kCycleStoreSizes
1. :red_square: kSelectors
1. :green_square: kUsableRows
1. :red_square: kPinnedVerifyingKey
1. :red_square: kTranscriptRepr
1. :blue_square: kLFirst
1. :blue_square: kLLast
1. :blue_square: kLActiveRow
1. :blue_square: kFixedColumns
1. :blue_square: kFixedPolys
1. :blue_square: kPermutationsColumns
1. :blue_square: kPermutationsPolys
1. :red_square: kProof
1. :blue_square: kAdviceCommitments
1. :blue_square: kChallenges
1. :red_square: kTheta
1. :blue_square: kLookupPermutedCommitmentsInput
1. :blue_square: kLookupPermutedCommitmentsTable
1. :blue_square: kLookupMPolyCommitments
1. :red_square: kBeta
1. :red_square: kGamma
1. :blue_square: kPermutationProductCommitments
1. :blue_square: kLookupProductCommitments
1. :blue_square: kLookupSumCommitments
1. :blue_square: kShuffleProductCommitments
1. :red_square: kY
1. :red_square: kVanishingHPolyCommitments
1. :red_square: kX
1. :blue_square: kAdviceEvals
1. :blue_square: kFixedEvals
1. :blue_square: kCommonPermutationEvals
1. :blue_square: kPermutationProductEvals
1. :blue_square: kPermutationProductNextEvals
1. :blue_square: kPermutationProductLastEvals
1. :blue_square: kLookupProductEvals
1. :blue_square: kLookupSumEvals
1. :blue_square: kLookupProductNextEvals
1. :blue_square: kLookupSumNextEvals
1. :blue_square: kLookupPermutedInputEvals
1. :blue_square: kLookupPermutedInputPrevEvals
1. :blue_square: kLookupPermutedTableEvals
1. :blue_square: kLookupMEvals
1. :blue_square: kShuffleProductEvals
1. :blue_square: kShuffleProductNextEvals
1. :red_square: kHEval

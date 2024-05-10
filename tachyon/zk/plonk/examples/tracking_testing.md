# Tracking Testing Types Across Example Circuits

## Currently Tested Types

| Circuit             | FloorPlanner       | PCS     | LS      |
| ------------------- | ------------------ | ------- | ------- |
| SimpleCircuit       | SimpleFloorPlanner | SHPlonk | Halo2LS |
| SimpleCircuit       | V1FloorPlanner     | SHPlonk | Halo2LS |
| SimpleLookupCircuit | SimpleFloorPlanner | SHPlonk | Halo2LS |
| SimpleLookupCircuit | V1FloorPlanner     | SHPlonk | Halo2LS |
| ShuffleCircuit      | SimpleFloorPlanner | SHPlonk | Halo2LS |
| ShuffleCircuit      | V1FloorPlanner     | SHPlonk | Halo2LS |

**In Progress**:

| Circuit        | FloorPlanner       | PCS | LS      |
| -------------- | ------------------ | --- | ------- |
| ShuffleCircuit | SimpleFloorPlanner | GWC | Halo2LS |
| ShuffleCircuit | V1FloorPlanner     | GWC | Halo2LS |

## Identical Test Data

The following circuits result in the same test data when using `SimpleFloorPlanner` or `V1FloorPlanner`

- SimpleLookupCircuit
- ShuffleCircuit

## Test Data Variables

:red_square: Required variables

:blue_square: Optional variables with flags

:green_square: Optional variables without flags (defaulted)

Refer to the following files to learn more about these variables:

- [circuit_test_data.h](circuit_test_data.h)
- [circuit_test.cc](circuit_test_data.cc)
- Any `...test_data.h` file:
  - [shuffle_circuit_test_data.h](shuffle_circuit_test_data.h)
  - [simple_circuit_test_data.h](simple_circuit_test_data.h)
  - [simple_lookup_circuit_test_data.h](simple_lookup_circuit_test_data.h)

1. :red_square: kN
2. :red_square: kPinnedConstraintSystem
3. :blue_square: kAssemblyFixedColumns
4. :blue_square: kAssemblyColumns
5. :blue_square: kCycleStoreMapping
6. :blue_square: kCycleStoreAux
7. :blue_square: kCycleStoreSizes
8. :red_square: kCycleStoreSelectors
9. :green_square: kUsableRows
10. :red_square: kPinnedVerifyingKey
11. :red_square: kTranscriptRepr
12. :blue_square: kLFirst
13. :blue_square: kLLast
14. :blue_square: kLActiveRow
15. :blue_square: kFixedColumns
16. :blue_square: kFixedPolys
17. :blue_square: kPermutationsColumns
18. :blue_square: kPermutationsPolys
19. :red_square: kProof
20. :blue_square: kAdviceCommitments
21. :blue_square: kChallenges
22. :red_square: kTheta
23. :blue_square: kPermutationProductCommitmentsPoints
24. :red_square: kBeta
25. :red_square: kGamma
26. :blue_square: kPermutationProductCommitments
27. :blue_square: kLookupProductCommitments
28. :red_square: kY
29. :blue_square: kVanishingHPolyCommitments
30. :red_square: kX
31. :blue_square: kAdviceEvals
32. :blue_square: kFixedEvals
33. :blue_square: kCommonPermutationEvals
34. :blue_square: kPermutationProductEvals
35. :blue_square: kPermutationProductNextEvals
36. :blue_square: kPermutationProductLastEvals
37. :blue_square: kLookupProductEvals
38. :blue_square: kLookupProductNextEvals
39. :blue_square: kLookupPermutedInputEvals
40. :blue_square: kLookupPermutedInputPrevEvals
41. :blue_square: kLookupPermutedTableEvals
42. :blue_square: kHEval

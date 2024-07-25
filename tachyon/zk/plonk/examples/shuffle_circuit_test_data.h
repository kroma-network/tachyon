#ifndef TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

#include <string_view>
#include <vector>

#include "tachyon/zk/plonk/examples/circuit_test_data.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PCS, typename LS, typename SFINAE = void>
class ShuffleTestData : public CircuitTestData<Circuit, PCS, LS> {};

// PCS = SHPlonk
template <typename Circuit, typename PCS, typename LS>
class ShuffleTestData<Circuit, PCS, LS, std::enable_if_t<IsSHPlonk<PCS>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;

  // Set flags of values to be used as true
  constexpr static bool kLFirstFlag = true;
  constexpr static bool kLLastFlag = true;
  constexpr static bool kLActiveRowFlag = true;
  constexpr static bool kFixedColumnsFlag = true;
  constexpr static bool kFixedPolysFlag = true;
  constexpr static bool kAdviceCommitmentsFlag = true;
  constexpr static bool kChallengesFlag = true;
  constexpr static bool kVanishingHPolyCommitmentsFlag = true;
  constexpr static bool kAdviceEvalsFlag = true;
  constexpr static bool kFixedEvalsFlag = true;

  constexpr static size_t kW = 2;
  constexpr static size_t kH = 8;
  constexpr static size_t kN = 16;

  // clang-format off
  constexpr static const std::string_view kOriginalTables[2][kW][kH] = {
    {
      {
        "0x0330fa29c0b79377aa26b9f89ad6c94912201b4c8a854c4fe1db0aae5d3e3139",
        "0x2b53081c28517939f3f3c8dd4feaa4d05c3fc2247814278abe7444937d23b649",
        "0x0435da43504bfe221f7408f38b0528a188e7534714de220af99504627dac267a",
        "0x0db560c95c6a428448974d4d1930d8e92bca484d0d944c0b5912fea74d2d2d14",
        "0x110f1d954c5579a04df83ad01a4cdaa4a0fe840ad909638d28f14d50fd9c688a",
        "0x23037b83210a774ff6b6caa3093f1c3a075ecc20f46ba78ce33f71214797853d",
        "0x003c96b2d19866aca8a6d299763cbfbe8d1e83063a24c1c64e2107bedc1d727e",
        "0x232a5cd6b5a054cf30ce5975952e9154721c0719a8584d8a6f6a90e3e5f1adc1",
      },
      {
        "0x263430d1d95ce2c8ee369a2d05b95564a3d00e58c54a6d45388a234afe13a9ce",
        "0x01fa5bd8e153542b410e3798d747c3ea06bf35d3f41ba326b1248d7936584099",
        "0x1030be57544d86a69777762fc9f8dd36a310c6aad17f74bf3c3631fc69d984c1",
        "0x093a3759811b856dfe2aae65b68b80014b6ab208ee942a153e8f1119189a0fc1",
        "0x06e6c64cffaf685871aa4c59dfe409c52dc6380286e234981afb5c36cf28acea",
        "0x18904ab2392340c3ec6230ccd1dcad9ec130f5a7d25306f2dd42d396e0e67fd7",
        "0x27eccd2de304de9dfef4d88b6f076a2c04937442e3e2c35b1f10d9c3fbb6325c",
        "0x1a24d928cc3f0ba998fec5b9b7f4e2a19a1853b75e7628beea1bd799be7f701a",
      },
    },
    {
      {
        "0x035cba522d844860cb05f47894a56cd1f6e2b3ec50ef96eab264d5f60b05a7f0",
        "0x2fe274ca9f0e7482712083b53697c1bb312ba37685e29df08ed9d4145076b959",
        "0x05ca447eb6d818c0df8ce00b2681700ca6e2f62e0e40f2eacef881b0ad16ceb0",
        "0x0aa9b6011fdae0573e6ec99ccd133b5434d23511c52bf68667d7a71610f24c1b",
        "0x028337e87c1e1d9833aa80d00571f46e0efe87447b73a7ed834dfc2c47e94e02",
        "0x00d025d0d32d6870a245b5f51e7dde6918431e91efc76ae50ceb9cb37a61cef9",
        "0x23ef42b9ae0dc781f8aa389a1388510de43df90c4ea1ed3d1d40811f4bca7015",
        "0x10f2cf0392479514a4f7c14f39483c799889a1045c1dc993c5b4148334e6694e",
      },
      {
        "0x035ad99713a98bea3f59b7dc1278c39718eed214329fccce30d70f78d6244815",
        "0x1155282d4492fdcd980f6c8a4507b0050ce36e1368bcdb15af11c42c3fe8a785",
        "0x248a66e7048c66a373945d6f4e57b2fd3cbb6324d606764c6f399edb5225ee36",
        "0x0c39a645b4354e23d8cad1e435f1e67ea351f18552b95988a3d817e2d22c81d0",
        "0x1237421269fe6491e84fbce500d23a57207bdd691ef35c94d378b97f82b16a3a",
        "0x00593f27ddf3f01b11d7556a8c5d187f0f9818873ad6f41696525969e799a9d7",
        "0x20cbdd95eda99ec21ccc683a973bd161416323be059c86187aae4f2d4399be9a",
        "0x131eef59955630cccce68f7f4d4a6a77a4d3831144a2eac1797561c571e67236",
      },
    }
  };

  constexpr static const std::string_view kShuffledTables[2][kW][kH] = {
    {
      {
        "0x23037b83210a774ff6b6caa3093f1c3a075ecc20f46ba78ce33f71214797853d",
        "0x0435da43504bfe221f7408f38b0528a188e7534714de220af99504627dac267a",
        "0x003c96b2d19866aca8a6d299763cbfbe8d1e83063a24c1c64e2107bedc1d727e",
        "0x2b53081c28517939f3f3c8dd4feaa4d05c3fc2247814278abe7444937d23b649",
        "0x232a5cd6b5a054cf30ce5975952e9154721c0719a8584d8a6f6a90e3e5f1adc1",
        "0x110f1d954c5579a04df83ad01a4cdaa4a0fe840ad909638d28f14d50fd9c688a",
        "0x0330fa29c0b79377aa26b9f89ad6c94912201b4c8a854c4fe1db0aae5d3e3139",
        "0x0db560c95c6a428448974d4d1930d8e92bca484d0d944c0b5912fea74d2d2d14",
      },
      {
        "0x18904ab2392340c3ec6230ccd1dcad9ec130f5a7d25306f2dd42d396e0e67fd7",
        "0x1030be57544d86a69777762fc9f8dd36a310c6aad17f74bf3c3631fc69d984c1",
        "0x27eccd2de304de9dfef4d88b6f076a2c04937442e3e2c35b1f10d9c3fbb6325c",
        "0x01fa5bd8e153542b410e3798d747c3ea06bf35d3f41ba326b1248d7936584099",
        "0x1a24d928cc3f0ba998fec5b9b7f4e2a19a1853b75e7628beea1bd799be7f701a",
        "0x06e6c64cffaf685871aa4c59dfe409c52dc6380286e234981afb5c36cf28acea",
        "0x263430d1d95ce2c8ee369a2d05b95564a3d00e58c54a6d45388a234afe13a9ce",
        "0x093a3759811b856dfe2aae65b68b80014b6ab208ee942a153e8f1119189a0fc1",
      },
    },
    {
      {
        "0x028337e87c1e1d9833aa80d00571f46e0efe87447b73a7ed834dfc2c47e94e02",
        "0x05ca447eb6d818c0df8ce00b2681700ca6e2f62e0e40f2eacef881b0ad16ceb0",
        "0x035cba522d844860cb05f47894a56cd1f6e2b3ec50ef96eab264d5f60b05a7f0",
        "0x10f2cf0392479514a4f7c14f39483c799889a1045c1dc993c5b4148334e6694e",
        "0x00d025d0d32d6870a245b5f51e7dde6918431e91efc76ae50ceb9cb37a61cef9",
        "0x23ef42b9ae0dc781f8aa389a1388510de43df90c4ea1ed3d1d40811f4bca7015",
        "0x0aa9b6011fdae0573e6ec99ccd133b5434d23511c52bf68667d7a71610f24c1b",
        "0x2fe274ca9f0e7482712083b53697c1bb312ba37685e29df08ed9d4145076b959",
      },
      {
        "0x1237421269fe6491e84fbce500d23a57207bdd691ef35c94d378b97f82b16a3a",
        "0x248a66e7048c66a373945d6f4e57b2fd3cbb6324d606764c6f399edb5225ee36",
        "0x035ad99713a98bea3f59b7dc1278c39718eed214329fccce30d70f78d6244815",
        "0x131eef59955630cccce68f7f4d4a6a77a4d3831144a2eac1797561c571e67236",
        "0x00593f27ddf3f01b11d7556a8c5d187f0f9818873ad6f41696525969e799a9d7",
        "0x20cbdd95eda99ec21ccc683a973bd161416323be059c86187aae4f2d4399be9a",
        "0x0c39a645b4354e23d8cad1e435f1e67ea351f18552b95988a3d817e2d22c81d0",
        "0x1155282d4492fdcd980f6c8a4507b0050ce36e1368bcdb15af11c42c3fe8a785",
      },
    },
  };
  // clang-format on

  // clang-format off
  constexpr static std::string_view kPinnedConstraintSystem =
      "PinnedConstraintSystem { "
        "num_fixed_columns: 0, "
        "num_advice_columns: 5, "
        "num_instance_columns: 0, "
        "num_selectors: 3, "
        "num_challenges: 2, "
        "advice_column_phase: ["
          "Phase(0), "
          "Phase(0), "
          "Phase(0), "
          "Phase(0), "
          "Phase(1)"
        "], "
        "challenge_phase: ["
          "Phase(0), "
          "Phase(0)"
        "], "
        "gates: ["
          "Product("
            "Selector(Selector(1, true)), "
            "Sum("
              "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
              "Negated(Advice { "
                "query_index: 0, "
                "column_index: 4, "
                "rotation: Rotation(0), "
                "phase: Phase(1) "
              "})"
            ")"
          "), "
          "Product("
            "Selector(Selector(2, true)), "
            "Sum("
              "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
              "Negated(Advice { "
                "query_index: 0, "
                "column_index: 4, "
                "rotation: Rotation(0), "
                "phase: Phase(1) "
              "})"
            ")"
          "), "
          "Product("
            "Selector(Selector(0, true)), "
            "Sum("
              "Product("
                "Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "}, "
                "Sum("
                  "Sum("
                    "Product("
                      "Advice { "
                        "query_index: 1, "
                        "column_index: 0, "
                        "rotation: Rotation(0) "
                      "}, "
                      "Challenge(Challenge { "
                        "index: 0, "
                        "phase: Phase(0) "
                      "})"
                    "), "
                    "Advice { "
                      "query_index: 2, "
                      "column_index: 1, "
                      "rotation: Rotation(0) "
                    "}"
                  "), "
                  "Challenge(Challenge { "
                    "index: 1, "
                    "phase: Phase(0) "
                  "})"
                ")"
              "), "
              "Negated(Product("
                "Advice { "
                  "query_index: 5, "
                  "column_index: 4, "
                  "rotation: Rotation(1), "
                  "phase: Phase(1) "
                "}, "
                "Sum("
                  "Sum("
                    "Product("
                      "Advice { "
                        "query_index: 3, "
                        "column_index: 2, "
                        "rotation: Rotation(0) "
                      "}, "
                      "Challenge(Challenge { "
                        "index: 0, "
                        "phase: Phase(0) "
                      "})"
                    "), "
                    "Advice { "
                      "query_index: 4, "
                      "column_index: 3, "
                      "rotation: Rotation(0) "
                    "}"
                  "), "
                  "Challenge(Challenge { "
                    "index: 1, "
                    "phase: Phase(0) "
                  "})"
                ")"
              "))"
            ")"
          ")"
        "], "
        "advice_queries: ["
          "("
            "Column { "
              "index: 4, "
              "column_type: Advice { "
                "phase: Phase(1) "
              "} "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 3, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 4, "
              "column_type: Advice { "
                "phase: Phase(1) "
              "} "
            "}, "
            "Rotation(1)"
          ")"
        "], "
        "instance_queries: [], "
        "fixed_queries: [], "
        "permutation: Argument { "
          "columns: [] "
        "}, "
        "lookups: [], "
        "constants: [], "
        "minimum_degree: None "
      "}";
  // clang-format on

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true, true, true, true, true, true, true, true,
        false, false, false, false, false, false, false, false},
      {true, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false},
      {false, false, false, false, false, false, false, false,
        true, false, false, false, false, false, false, false},
  };
  // clang-format on

  // clang-format off
  constexpr static std::string_view kPinnedVerifyingKey =
      "PinnedVerificationKey { "
        "base_modulus: \"0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47\", "
        "scalar_modulus: \"0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001\", "
        "domain: PinnedEvaluationDomain { "
          "k: 4, "
          "extended_k: 5, "
          "omega: 0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b "
        "}, "
        "cs: PinnedConstraintSystem { "
          "num_fixed_columns: 2, "
          "num_advice_columns: 5, "
          "num_instance_columns: 0, "
          "num_selectors: 3, "
          "num_challenges: 2, "
          "advice_column_phase: ["
            "Phase(0), "
            "Phase(0), "
            "Phase(0), "
            "Phase(0), "
            "Phase(1)"
          "], "
          "challenge_phase: ["
            "Phase(0), "
            "Phase(0)"
          "], "
          "gates: ["
            "Product("
              "Product("
                "Fixed { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}, "
                "Sum("
                  "Constant(0x0000000000000000000000000000000000000000000000000000000000000002), "
                  "Negated(Fixed { "
                    "query_index: 1, "
                    "column_index: 1, "
                    "rotation: Rotation(0) "
                  "})"
                ")"
              "), "
              "Sum("
                "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                "Negated(Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "})"
              ")"
            "), "
            "Product("
              "Product("
                "Fixed { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}, "
                "Sum("
                  "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                  "Negated(Fixed { "
                    "query_index: 1, "
                    "column_index: 1, "
                    "rotation: Rotation(0) "
                  "})"
                ")"
              "), "
              "Sum("
                "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                "Negated(Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "})"
              ")"
            "), "
            "Product("
              "Fixed { "
                "query_index: 0, "
                "column_index: 0, "
                "rotation: Rotation(0) "
              "}, "
              "Sum("
                "Product("
                  "Advice { "
                    "query_index: 0, "
                    "column_index: 4, "
                    "rotation: Rotation(0), "
                    "phase: Phase(1) "
                  "}, "
                  "Sum("
                    "Sum("
                      "Product("
                        "Advice { "
                          "query_index: 1, "
                          "column_index: 0, "
                          "rotation: Rotation(0) "
                        "}, "
                        "Challenge(Challenge { "
                          "index: 0, "
                          "phase: Phase(0) "
                        "})"
                      "), "
                      "Advice { "
                        "query_index: 2, "
                        "column_index: 1, "
                        "rotation: Rotation(0) "
                      "}"
                    "), "
                    "Challenge(Challenge { "
                      "index: 1, "
                      "phase: Phase(0) "
                    "})"
                  ")"
                "), "
                "Negated(Product("
                  "Advice { "
                    "query_index: 5, "
                    "column_index: 4, "
                    "rotation: Rotation(1), "
                    "phase: Phase(1) "
                  "}, "
                  "Sum("
                    "Sum("
                      "Product("
                        "Advice { "
                          "query_index: 3, "
                          "column_index: 2, "
                          "rotation: Rotation(0) "
                        "}, "
                        "Challenge(Challenge { "
                          "index: 0, "
                          "phase: Phase(0) "
                        "})"
                      "), "
                      "Advice { "
                        "query_index: 4, "
                        "column_index: 3, "
                        "rotation: Rotation(0) "
                      "}"
                    "), "
                    "Challenge(Challenge { "
                      "index: 1, "
                      "phase: Phase(0) "
                    "})"
                  ")"
                "))"
              ")"
            ")"
          "], "
          "advice_queries: ["
            "("
              "Column { "
                "index: 4, "
                "column_type: Advice { "
                  "phase: Phase(1) "
                "} "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 1, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 2, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 3, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 4, "
                "column_type: Advice { "
                  "phase: Phase(1) "
                "} "
              "}, "
              "Rotation(1)"
            ")"
          "], "
          "instance_queries: [], "
          "fixed_queries: ["
            "("
              "Column { "
                "index: 0, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 1, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            ")"
          "], "
          "permutation: Argument { "
            "columns: [] "
          "}, "
          "lookups: [], "
          "constants: [], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x297ff8a661d1fa1196c065b6fb7df901fb16b5b83168ddca6c7749d963cc967a, "
            "0x1a7b1f2b5f3e35fc4c706ece6b8f646e95904f9aa417f8a31fd37a41da167ec1), "
          "(0x1025d577f7527c7c9a5d132164beef9ec3ebc63805b146843466f6cdf5fb37d4, "
            "0x1ba406f864ec730f7f725b3478ce1e22c5579b6228593c9ba36e9cb18a10aab3)"
        "], "
        "permutation: VerifyingKey { "
          "commitments: [] "
        "} "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x220eecfcd245ace6db06b0c892cef067bb85cc36dc0401c6e05b618bb939f761";

  constexpr static std::string_view kLFirst[] = {
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
  };

  constexpr static std::string_view kLLast[] = {
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2014447de15a99b6df03833e95f96ae1299c9ec6ff990b6e75fa3b3b04846a57",
      "0x0f1f5883e65f820d14d56342dc92fd12a944d4cbbdce5377b7439bd07108fc9d",
      "0x02b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e8",
      "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
      "0x105009f4ffd70672d94cc277eb87ed7bfe9749817a206522cde7ba58eb7b95aa",
      "0x2144f5eefad21e1ca37ae273a4ee5b4a7eef137cbbeb1d198c9e59c37ef70364",
      "0x2db11694c4a58b3789868bd388165969c30dc1120a37fff09a991abf43e42a19",
      "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
      "0x2014447de15a99b6df03833e95f96ae1299c9ec6ff990b6e75fa3b3b04846a57",
      "0x0f1f5883e65f820d14d56342dc92fd12a944d4cbbdce5377b7439bd07108fc9d",
      "0x02b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e8",
      "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
      "0x105009f4ffd70672d94cc277eb87ed7bfe9749817a206522cde7ba58eb7b95aa",
      "0x2144f5eefad21e1ca37ae273a4ee5b4a7eef137cbbeb1d198c9e59c37ef70364",
      "0x2db11694c4a58b3789868bd388165969c30dc1120a37fff09a991abf43e42a19",
  };

  constexpr static std::string_view kLActiveRow[] = {
      "0x12259d6b14729c0fa51e1a2470908122ef13771b2da58a367974bc177a000001",
      "0x1a8133201ba2fe22aff30c7fd04fadcb17ceab0715a13371b06d35acd9695b3d",
      "0x0d49c50dd1c3ec703dc7be1c836fd7f62c140afcf284ce19b9a99affac7b95aa",
      "0x117dae38a05bcb8c7c290ee16cec493d73ef8dafa5c61fa4a2efd9a39e63abf4",
      "0x0c19139cb84c680a79505ee7747ae78cd6c196473632bc6ea3057c773208fc9d",
      "0x136156e428a2662bc2fddfd3b39f6475dafecb8699a611f3da6edd22c3479af1",
      "0x2aaad1ad96927134ee0187781ffe43e3f08a828d829c68e7865afb6604e42a19",
      "0x28d7d254c17b7ea40ebc4659996adacebd0d8f52d021284040d407c2f33b896f",
      "0x0000000000000000000000000000000000000000000000000000000000000000",
      "0x1ef7221577b1e8f8cebdcb2fcd10296b6d9d43267b395c820757c0fc87681d81",
      "0x1d0dff96b3477fb4437e7ee32de1555b5719604277fd746561bc1be1c5846a57",
      "0x1be193603aacb384678281793fa8f20b949cf5976d9493017ba8a357ee49da57",
      "0x1e3eb107ccbf041a07f5de183cd645c4ac6bd4f8344f861078603a6a3ff70364",
      "0x20080468beb85b16c9f71b3ea2ce10fb6cdc81c346721c888ebc9109900adec6",
      "0x30114169cfaa9b194b94fb3e12d441cabad6d0fa619f4a28d8ecb10f5d1bd5e9",
      "0x2edcc3ce4ec47abd9b83b31a4db9571236223b590c30997fd30ce24f7bf2fdd6",
  };

  constexpr static std::string_view kFixedColumns[][kN] = {
      {
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
      {
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000002",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
  };

  constexpr static std::string_view kFixedPolys[][kN] = {
      {
          "0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000001",
          "0x1db5b8e23dbf6d914e37d939eb6b037c619b3e5ea827cdb2b030fc247bdc9dcb",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x15612923e039d29887b96603ef6ab434a2c021b8674489fcfa35d3972944e837",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x186635fde662f3b15c5e1b5fc456d148c26786b9cda347e81f7f4c2f8c9761ed",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x14a1b21f8b9c269f87d74740cf8c84a3046338799106ae503c4dfb0f0b0b250e",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x15b61284f96f4584f96ef5bee1c4a8ae7eca32c5d97b942edf17bbd266f4daf3",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x11f18ea69ea8787324e8219fecfa5c08c0c5e4859cdefa96fbe66ab1e5689e14",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x14f69b80a4d1998bf98cd6fbc1e6791ce06d4987033db882212fe34a48bb17ca",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0ca20bc2474bfe93330e63c5c5e629d521922ce0c25a74cc6b34babcf6236236",
      },
      {
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x030644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000",
      },
  };

  constexpr static uint8_t kProof[] = {
      122, 117, 229, 163, 114, 52,  159, 223, 87,  191, 210, 4,   56,  250,
      59,  18,  206, 131, 78,  129, 60,  202, 130, 174, 203, 80,  0,   221,
      190, 150, 199, 170, 97,  250, 239, 195, 165, 188, 251, 230, 129, 105,
      212, 209, 41,  124, 75,  202, 166, 142, 57,  16,  160, 164, 175, 119,
      80,  96,  210, 1,   220, 55,  229, 150, 176, 0,   127, 136, 63,  27,
      146, 170, 55,  152, 223, 250, 131, 165, 92,  170, 126, 214, 34,  187,
      172, 236, 248, 250, 195, 21,  121, 118, 69,  108, 167, 132, 242, 42,
      222, 129, 23,  144, 206, 50,  2,   128, 3,   44,  222, 217, 223, 35,
      228, 184, 246, 76,  34,  146, 170, 5,   127, 243, 140, 134, 59,  224,
      63,  139, 97,  91,  176, 229, 208, 22,  81,  76,  102, 212, 230, 42,
      248, 115, 0,   57,  161, 155, 76,  46,  222, 63,  50,  191, 169, 117,
      109, 77,  100, 113, 115, 163, 219, 109, 33,  128, 52,  174, 106, 188,
      197, 159, 82,  64,  13,  77,  214, 88,  94,  3,   59,  239, 242, 15,
      57,  52,  184, 96,  8,   195, 219, 8,   28,  165, 23,  155, 225, 52,
      2,   100, 158, 139, 3,   255, 216, 141, 144, 92,  148, 43,  245, 117,
      105, 244, 203, 45,  169, 199, 189, 108, 165, 254, 123, 192, 88,  166,
      183, 219, 170, 20,  145, 67,  54,  188, 84,  208, 144, 229, 73,  102,
      239, 199, 192, 67,  84,  125, 77,  150, 208, 152, 34,  245, 28,  245,
      53,  241, 240, 142, 238, 116, 44,  171, 9,   62,  209, 106, 14,  52,
      209, 60,  194, 175, 33,  157, 149, 131, 25,  211, 81,  22,  167, 97,
      84,  92,  7,   120, 171, 195, 156, 29,  78,  254, 1,   92,  9,   249,
      182, 255, 222, 133, 161, 215, 117, 56,  201, 12,  65,  236, 138, 136,
      207, 214, 29,  1,   218, 176, 164, 120, 50,  120, 130, 42,  1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   96,  134, 180, 185, 62,  13,  187, 73,  198, 194, 240, 58,
      143, 101, 182, 194, 201, 129, 242, 182, 19,  7,   12,  75,  34,  92,
      65,  156, 85,  241, 183, 171, 45,  201, 146, 14,  118, 58,  117, 27,
      61,  108, 249, 137, 144, 71,  115, 19,  73,  98,  4,   77,  237, 52,
      69,  171, 85,  201, 246, 215, 91,  71,  145, 175, 139, 126, 48,  67,
      66,  130, 140, 51,  253, 115, 220, 250, 206, 22,  138, 113, 63,  0,
      74,  233, 167, 144, 210, 243, 212, 44,  42,  58,  221, 5,   215, 30,
      94,  146, 3,   134, 1,   201, 167, 90,  132, 197, 56,  187, 221, 23,
      242, 164, 87,  157, 142, 157, 80,  235, 12,  136, 21,  115, 181, 19,
      96,  199, 209, 5,   135, 209, 93,  98,  16,  78,  106, 217, 160, 233,
      35,  255, 207, 251, 154, 71,  20,  253, 239, 202, 139, 237, 246, 171,
      137, 255, 94,  92,  53,  190, 236, 34,  153, 4,   94,  63,  214, 40,
      181, 15,  147, 150, 81,  110, 181, 99,  38,  34,  56,  239, 28,  121,
      151, 11,  251, 82,  30,  105, 132, 33,  69,  126, 142, 15,  157, 153,
      218, 141, 253, 210, 254, 249, 230, 130, 27,  49,  161, 88,  161, 222,
      147, 49,  245, 92,  45,  16,  131, 95,  217, 194, 215, 16,  69,  71,
      78,  29,  108, 202, 1,   68,  66,  44,  39,  172, 207, 144, 77,  121,
      120, 114, 211, 147, 52,  200, 152, 197, 161, 235, 76,  139, 115, 141,
      206, 208, 134, 251, 78,  15,  16,  93,  166, 218, 134, 85,  60,  230,
      36,  206, 42,  75,  206, 2,   145, 43,  18,  145, 135, 107, 152, 229,
      5,   62,  206, 252, 129, 232, 114, 75,  228, 33,  140, 218, 27,  139,
      246, 80,  127, 36,  80,  115, 177, 219, 69,  80,  235, 123, 191, 251,
      189, 206, 103, 248, 242, 58,  173, 185, 101, 112, 138, 176, 235, 34,
      81,  60,  31,  132, 241, 220, 231, 212, 135, 74,  159, 124, 115, 227,
      149, 192, 56,  152, 43,  182, 158, 246, 63,  125, 148, 144, 243, 217,
      66,  119, 14,  38,  238, 129, 238, 25,  126, 177, 4,   135, 84,  255,
      216, 141, 221, 137, 172, 22,  182, 33,  217, 8,   95,  92,  208, 83,
      226, 5,   67,  65,  218, 137, 253, 12,  231, 216, 174, 122, 162, 239,
      88,  16,  65,  125, 101, 123, 96,  188, 118, 164, 68,  145, 203, 129,
      54,  95,  84,  4,   83,  98,  53,  87,  145, 106, 223, 26,  146, 169,
      158, 159, 209, 157, 89,  102, 204, 254, 97,  92,  149, 104, 21,  94,
      238, 231, 74,  192, 45,  11,  167, 112, 183, 54,  189, 61,  176, 204,
      151, 2,   216, 147, 249, 148, 195, 213, 197, 49,  184, 7,   174, 45,
      175, 251, 144, 126, 180, 32,  145, 186, 94,  84,  170, 71,  228, 213,
      109, 89,  141, 161, 164, 38,  204, 163, 86,  138, 200, 222, 184, 232,
      163, 254, 172, 253, 164, 60,  8,   91,  31,  113, 46,  55,  14,  133,
      120, 246, 172, 208, 57,  35,  123, 179, 212, 32,  1,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      194, 213, 8,   179, 79,  153, 219, 29,  172, 38,  212, 118, 159, 250,
      149, 134, 98,  218, 34,  34,  95,  113, 0,   26,  133, 92,  226, 144,
      186, 56,  200, 41,  115, 243, 72,  67,  69,  61,  133, 160, 71,  211,
      240, 160, 254, 68,  206, 47,  27,  164, 205, 166, 188, 172, 123, 134,
      154, 35,  148, 250, 50,  159, 23,  168};

  // clang-format off
  constexpr static Point kAdviceCommitments[][5] = {
      {
          {"0x2ac796bedd0050cbae82ca3c814e83ce123bfa3804d2bf57df9f3472a3e5757a",
           "0x225c37b55b77d7c61ec1b3497b4cd49e88979a7529f69157359cfb31af4baa8d"},
          {"0x16e537dc01d2605077afa4a010398ea6ca4b7c29d1d46981e6fbbca5c3effa61",
           "0x274fdf6ec7a8f5080e28b30278cdc8993517cbae5f2b1d2d09c67e0be64a1419"},
          {"0x04a76c45767915c3faf8ecacbb22d67eaa5ca583fadf9837aa921b3f887f00b0",
           "0x03ed297fdc6199331c528b55c05c0fbf77a1afb96ce7fdc91d2ba74fcc8dac41"},
          {"0x0b3fe03b868cf37f05aa92224cf6b8e423dfd9de2c03800232ce901781de2af2",
           "0x1d3274f270c5b3969008d4e6b4aa75bf816b34a82ad2f01edf7c7b11138d05b5"},
          {"0x1d9cc3ab78075c5461a71651d31983959d21afc23cd1340e6ad13e09ab2c74ee",
           "0x01badd1509eebaca90b100b0fbfbad42486e4619f186715c71aaaa8f8217cfa2"},
      },
      {
          {"0x237371644d6d75a9bf323fde2e4c9ba1390073f82ae6d4664c5116d0e5b05b61",
           "0x046f8e3efa26d48fa674c14f6ed386b43503bd345fd9d8d81614723520c57761"},
          {"0x251c08dbc30860b834390ff2ef3b035e58d64d0d40529fc5bc6aae3480216ddb",
           "0x1af96eb77e286bd14427ad56cb681c9ba67716ae44e383b871b53b405b4db24b"},
          {"0x2658c07bfea56cbdc7a92dcbf46975f52b945c908dd8ff038b9e640234e19b17",
           "0x1faea45a2a8673c64b9e122a48724d8ca2978ba5fc7470b92851c5839fb9744b"},
          {"0x0ef0f135f51cf52298d0964d7d5443c0c7ef6649e590d054bc36439114aadbb7",
           "0x2688b891c83687257bfb32b08f072a4d876f652bcaaf673217dcfc8679d4ac39"},
          {"0x2a82783278a4b0da011dd6cf888aec410cc93875d7a185deffb6f9095c01fe4e",
           "0x00610abd37a882e9635be4ec3ab465ac92fa1c029392a7a16c69358c46432450"},
      },
  };
  // clang-format on

  constexpr static std::string_view kChallenges[] = {
      "0x2f2cdac7ef66f10d3c910f13e5a4344318eeeceae9edcdd75b4c1976de43475e",
      "0x154c67d8fa9170f88d424e7634e151902bcf91212c5f7192b58094f50f4da90c"};

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x2bb7f1559c415c224b0c0713b6f281c9c2b6658f3af0c2c649bb0d3eb9b48660",
       "0x0b4621657cbc27228adc9011edbd154e9b1b743a9901ce7587b5802c4b93e4a5"},
      {"0x2f91475bd7f6c955ab4534ed4d0462491373479089f96c3d1b753a760e92c92d",
       "0x0f2b30e0179380f1a9dda5cb59eb5ff6c0c093f39ff3129b09a8fb4493fcb719"},
  };

  constexpr static std::string_view kTheta =
      "0x1b05d778b77bda225d239dffc14f3976e8fe1760fd21713b610c3f213589400f";

  constexpr static std::string_view kBeta =
      "0x1a98a31b28f4089c8af7cdb48e9e056838679dffae6e96ff083cb8dba53a9143";

  constexpr static std::string_view kGamma =
      "0x2dc0d56a5db87e084ca72c1c51467b28d927ff94227126f04f69cb21965081f8";

  constexpr static std::string_view kY =
      "0x17e9a336d7b197716486c320c264d80a5eab37371602ad38a6feb22486894c17";

  constexpr static std::string_view kX =
      "0x14479ed68753bcc9dfd9c779bc02c61fefa774a94b9b9030c94383bdc4bf4c43";

  constexpr static std::string_view kAdviceEvals[][6] = {
      {
          "0x1ed705dd3a2a2cd4f3d290a7e94a003f718a16cefadc73fd338c824243307e8b",
          "0x05d1c76013b57315880ceb509d8e9d57a4f217ddbb38c5845aa7c9018603925e",
          "0x22ecbe355c5eff89abf6ed8bcaeffd14479afbcfff23e9a0d96a4e10625dd187",
          "0x0f8e7e452184691e52fb0b97791cef38222663b56e5196930fb528d63f5e0499",
          "0x1d4e474510d7c2d95f83102d5cf53193dea158a1311b82e6f9fed2fd8dda999d",
          "0x0f4efb86d0ce8d738b4ceba1c598c83493d37278794d90cfac272c424401ca6c",
      },
      {
          "0x21e44b72e881fcce3e05e5986b8791122b9102ce4b2ace24e63c5586daa65d10",
          "0x22ebb08a7065b9ad3af2f867cebdfbbf7beb5045dbb17350247f50f68b1bda8c",
          "0x260e7742d9f390947d3ff69eb62b9838c095e3737c9f4a87d4e7dcf1841f3c51",
          "0x0cfd89da414305e253d05c5f08d921b616ac89dd8dd8ff548704b17e19ee81ee",
          "0x1adf6a915735625304545f3681cb9144a476bc607b657d411058efa27aaed8e7",
          "0x0297ccb03dbd36b770a70b2dc04ae7ee5e1568955c61fecc66599dd19f9ea992",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x26a4a18d596dd5e447aa545eba9120b47e90fbaf2dae07b831c5d5c394f993d8",
      "0x20d4b37b2339d0acf678850e372e711f5b083ca4fdacfea3e8b8dec88a56a3cc",
  };

  constexpr static std::string_view kHEval =
      "0x0cdce47f9c4b0dfea0fffc3e184b36a02517c68ecae1d32921c9050771799bd2";

  static void TestConfig(ShuffleCircuitConfig<F, kW>& config) {
    EXPECT_EQ(config.q_shuffle(), Selector::Simple(0));
    EXPECT_EQ(config.q_first(), Selector::Simple(1));
    EXPECT_EQ(config.q_last(), Selector::Simple(2));
    for (size_t i = 0; i < kW; i++) {
      EXPECT_EQ(config.original_column_keys()[i], AdviceColumnKey(i));
      EXPECT_EQ(config.shuffled_column_keys()[i], AdviceColumnKey(kW + i));
    }
    EXPECT_EQ(config.theta(), Challenge(0, kFirstPhase));
    EXPECT_EQ(config.gamma(), Challenge(1, kFirstPhase));
    EXPECT_EQ(config.z(), AdviceColumnKey(2 * kW, kSecondPhase));
  }

  static Circuit GetCircuit(size_t i = 0) {
    CHECK_LT(i, std::size(kOriginalTables));
    return {CreateTable(kOriginalTables[i]), CreateTable(kShuffledTables[i])};
  }

  static std::vector<Circuit> Get2Circuits() {
    return {GetCircuit(0), GetCircuit(1)};
  }

 private:
  static std::vector<std::vector<F>> CreateTable(
      const std::string_view table[kW][kH]) {
    return base::CreateVector(kW, [table](size_t i) {
      return base::CreateVector(
          kH, [table, i](size_t j) { return *F::FromHexString(table[i][j]); });
    });
  }
};

// TODO(ashjeong): Obtain all data for the GWC version and run
// PCS = GWC
template <typename Circuit, typename PCS, typename LS>
class ShuffleTestData<Circuit, PCS, LS, std::enable_if_t<IsGWC<PCS>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;

  constexpr static size_t kW = 2;
  constexpr static size_t kH = 8;
  constexpr static size_t kN = 16;

  // clang-format off
  constexpr static const std::string_view kOriginalTables[2][kW][kH] = {
    {
      {
        "0x0330fa29c0b79377aa26b9f89ad6c94912201b4c8a854c4fe1db0aae5d3e3139",
        "0x2b53081c28517939f3f3c8dd4feaa4d05c3fc2247814278abe7444937d23b649",
        "0x0435da43504bfe221f7408f38b0528a188e7534714de220af99504627dac267a",
        "0x0db560c95c6a428448974d4d1930d8e92bca484d0d944c0b5912fea74d2d2d14",
        "0x110f1d954c5579a04df83ad01a4cdaa4a0fe840ad909638d28f14d50fd9c688a",
        "0x23037b83210a774ff6b6caa3093f1c3a075ecc20f46ba78ce33f71214797853d",
        "0x003c96b2d19866aca8a6d299763cbfbe8d1e83063a24c1c64e2107bedc1d727e",
        "0x232a5cd6b5a054cf30ce5975952e9154721c0719a8584d8a6f6a90e3e5f1adc1",
      },
      {
        "0x263430d1d95ce2c8ee369a2d05b95564a3d00e58c54a6d45388a234afe13a9ce",
        "0x01fa5bd8e153542b410e3798d747c3ea06bf35d3f41ba326b1248d7936584099",
        "0x1030be57544d86a69777762fc9f8dd36a310c6aad17f74bf3c3631fc69d984c1",
        "0x093a3759811b856dfe2aae65b68b80014b6ab208ee942a153e8f1119189a0fc1",
        "0x06e6c64cffaf685871aa4c59dfe409c52dc6380286e234981afb5c36cf28acea",
        "0x18904ab2392340c3ec6230ccd1dcad9ec130f5a7d25306f2dd42d396e0e67fd7",
        "0x27eccd2de304de9dfef4d88b6f076a2c04937442e3e2c35b1f10d9c3fbb6325c",
        "0x1a24d928cc3f0ba998fec5b9b7f4e2a19a1853b75e7628beea1bd799be7f701a",
      },
    },
    {
      {
        "0x035cba522d844860cb05f47894a56cd1f6e2b3ec50ef96eab264d5f60b05a7f0",
        "0x2fe274ca9f0e7482712083b53697c1bb312ba37685e29df08ed9d4145076b959",
        "0x05ca447eb6d818c0df8ce00b2681700ca6e2f62e0e40f2eacef881b0ad16ceb0",
        "0x0aa9b6011fdae0573e6ec99ccd133b5434d23511c52bf68667d7a71610f24c1b",
        "0x028337e87c1e1d9833aa80d00571f46e0efe87447b73a7ed834dfc2c47e94e02",
        "0x00d025d0d32d6870a245b5f51e7dde6918431e91efc76ae50ceb9cb37a61cef9",
        "0x23ef42b9ae0dc781f8aa389a1388510de43df90c4ea1ed3d1d40811f4bca7015",
        "0x10f2cf0392479514a4f7c14f39483c799889a1045c1dc993c5b4148334e6694e",
      },
      {
        "0x035ad99713a98bea3f59b7dc1278c39718eed214329fccce30d70f78d6244815",
        "0x1155282d4492fdcd980f6c8a4507b0050ce36e1368bcdb15af11c42c3fe8a785",
        "0x248a66e7048c66a373945d6f4e57b2fd3cbb6324d606764c6f399edb5225ee36",
        "0x0c39a645b4354e23d8cad1e435f1e67ea351f18552b95988a3d817e2d22c81d0",
        "0x1237421269fe6491e84fbce500d23a57207bdd691ef35c94d378b97f82b16a3a",
        "0x00593f27ddf3f01b11d7556a8c5d187f0f9818873ad6f41696525969e799a9d7",
        "0x20cbdd95eda99ec21ccc683a973bd161416323be059c86187aae4f2d4399be9a",
        "0x131eef59955630cccce68f7f4d4a6a77a4d3831144a2eac1797561c571e67236",
      },
    }
  };

  constexpr static const std::string_view kShuffledTables[2][kW][kH] = {
    {
      {
        "0x23037b83210a774ff6b6caa3093f1c3a075ecc20f46ba78ce33f71214797853d",
        "0x0435da43504bfe221f7408f38b0528a188e7534714de220af99504627dac267a",
        "0x003c96b2d19866aca8a6d299763cbfbe8d1e83063a24c1c64e2107bedc1d727e",
        "0x2b53081c28517939f3f3c8dd4feaa4d05c3fc2247814278abe7444937d23b649",
        "0x232a5cd6b5a054cf30ce5975952e9154721c0719a8584d8a6f6a90e3e5f1adc1",
        "0x110f1d954c5579a04df83ad01a4cdaa4a0fe840ad909638d28f14d50fd9c688a",
        "0x0330fa29c0b79377aa26b9f89ad6c94912201b4c8a854c4fe1db0aae5d3e3139",
        "0x0db560c95c6a428448974d4d1930d8e92bca484d0d944c0b5912fea74d2d2d14",
      },
      {
        "0x18904ab2392340c3ec6230ccd1dcad9ec130f5a7d25306f2dd42d396e0e67fd7",
        "0x1030be57544d86a69777762fc9f8dd36a310c6aad17f74bf3c3631fc69d984c1",
        "0x27eccd2de304de9dfef4d88b6f076a2c04937442e3e2c35b1f10d9c3fbb6325c",
        "0x01fa5bd8e153542b410e3798d747c3ea06bf35d3f41ba326b1248d7936584099",
        "0x1a24d928cc3f0ba998fec5b9b7f4e2a19a1853b75e7628beea1bd799be7f701a",
        "0x06e6c64cffaf685871aa4c59dfe409c52dc6380286e234981afb5c36cf28acea",
        "0x263430d1d95ce2c8ee369a2d05b95564a3d00e58c54a6d45388a234afe13a9ce",
        "0x093a3759811b856dfe2aae65b68b80014b6ab208ee942a153e8f1119189a0fc1",
      },
    },
    {
      {
        "0x028337e87c1e1d9833aa80d00571f46e0efe87447b73a7ed834dfc2c47e94e02",
        "0x05ca447eb6d818c0df8ce00b2681700ca6e2f62e0e40f2eacef881b0ad16ceb0",
        "0x035cba522d844860cb05f47894a56cd1f6e2b3ec50ef96eab264d5f60b05a7f0",
        "0x10f2cf0392479514a4f7c14f39483c799889a1045c1dc993c5b4148334e6694e",
        "0x00d025d0d32d6870a245b5f51e7dde6918431e91efc76ae50ceb9cb37a61cef9",
        "0x23ef42b9ae0dc781f8aa389a1388510de43df90c4ea1ed3d1d40811f4bca7015",
        "0x0aa9b6011fdae0573e6ec99ccd133b5434d23511c52bf68667d7a71610f24c1b",
        "0x2fe274ca9f0e7482712083b53697c1bb312ba37685e29df08ed9d4145076b959",
      },
      {
        "0x1237421269fe6491e84fbce500d23a57207bdd691ef35c94d378b97f82b16a3a",
        "0x248a66e7048c66a373945d6f4e57b2fd3cbb6324d606764c6f399edb5225ee36",
        "0x035ad99713a98bea3f59b7dc1278c39718eed214329fccce30d70f78d6244815",
        "0x131eef59955630cccce68f7f4d4a6a77a4d3831144a2eac1797561c571e67236",
        "0x00593f27ddf3f01b11d7556a8c5d187f0f9818873ad6f41696525969e799a9d7",
        "0x20cbdd95eda99ec21ccc683a973bd161416323be059c86187aae4f2d4399be9a",
        "0x0c39a645b4354e23d8cad1e435f1e67ea351f18552b95988a3d817e2d22c81d0",
        "0x1155282d4492fdcd980f6c8a4507b0050ce36e1368bcdb15af11c42c3fe8a785",
      },
    },
  };
  // clang-format on

  // clang-format off
  constexpr static std::string_view kPinnedConstraintSystem =
      "PinnedConstraintSystem { "
        "num_fixed_columns: 0, "
        "num_advice_columns: 5, "
        "num_instance_columns: 0, "
        "num_selectors: 3, "
        "num_challenges: 2, "
        "advice_column_phase: ["
          "Phase(0), "
          "Phase(0), "
          "Phase(0), "
          "Phase(0), "
          "Phase(1)"
        "], "
        "challenge_phase: ["
          "Phase(0), "
          "Phase(0)"
        "], "
        "gates: ["
          "Product("
            "Selector(Selector(1, true)), "
            "Sum("
              "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
              "Negated(Advice { "
                "query_index: 0, "
                "column_index: 4, "
                "rotation: Rotation(0), "
                "phase: Phase(1) "
              "})"
            ")"
          "), "
          "Product("
            "Selector(Selector(2, true)), "
            "Sum("
              "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
              "Negated(Advice { "
                "query_index: 0, "
                "column_index: 4, "
                "rotation: Rotation(0), "
                "phase: Phase(1) "
              "})"
            ")"
          "), "
          "Product("
            "Selector(Selector(0, true)), "
            "Sum("
              "Product("
                "Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "}, "
                "Sum("
                  "Sum("
                    "Product("
                      "Advice { "
                        "query_index: 1, "
                        "column_index: 0, "
                        "rotation: Rotation(0) "
                      "}, "
                      "Challenge(Challenge { "
                        "index: 0, "
                        "phase: Phase(0) "
                      "})"
                    "), "
                    "Advice { "
                      "query_index: 2, "
                      "column_index: 1, "
                      "rotation: Rotation(0) "
                    "}"
                  "), "
                  "Challenge(Challenge { "
                    "index: 1, "
                    "phase: Phase(0) "
                  "})"
                ")"
              "), "
              "Negated(Product("
                "Advice { "
                  "query_index: 5, "
                  "column_index: 4, "
                  "rotation: Rotation(1), "
                  "phase: Phase(1) "
                "}, "
                "Sum("
                  "Sum("
                    "Product("
                      "Advice { "
                        "query_index: 3, "
                        "column_index: 2, "
                        "rotation: Rotation(0) "
                      "}, "
                      "Challenge(Challenge { "
                        "index: 0, "
                        "phase: Phase(0) "
                      "})"
                    "), "
                    "Advice { "
                      "query_index: 4, "
                      "column_index: 3, "
                      "rotation: Rotation(0) "
                    "}"
                  "), "
                  "Challenge(Challenge { "
                    "index: 1, "
                    "phase: Phase(0) "
                  "})"
                ")"
              "))"
            ")"
          ")"
        "], "
        "advice_queries: ["
          "("
            "Column { "
              "index: 4, "
              "column_type: Advice { "
                "phase: Phase(1) "
              "} "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 3, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          "), "
          "("
            "Column { "
              "index: 4, "
              "column_type: Advice { "
                "phase: Phase(1) "
              "} "
            "}, "
            "Rotation(1)"
          ")"
        "], "
        "instance_queries: [], "
        "fixed_queries: [], "
        "permutation: Argument { "
          "columns: [] "
        "}, "
        "lookups: [], "
        "constants: [], "
        "minimum_degree: None "
      "}";
  // clang-format on

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true, true, true, true, true, true, true, true,
        false, false, false, false, false, false, false, false},
      {true, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false},
      {false, false, false, false, false, false, false, false,
        true, false, false, false, false, false, false, false},
  };
  // clang-format on

  // clang-format off
  constexpr static std::string_view kPinnedVerifyingKey =
      "PinnedVerificationKey { "
        "base_modulus: \"0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47\", "
        "scalar_modulus: \"0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001\", "
        "domain: PinnedEvaluationDomain { "
          "k: 4, "
          "extended_k: 5, "
          "omega: 0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b "
        "}, "
        "cs: PinnedConstraintSystem { "
          "num_fixed_columns: 2, "
          "num_advice_columns: 5, "
          "num_instance_columns: 0, "
          "num_selectors: 3, "
          "num_challenges: 2, "
          "advice_column_phase: ["
            "Phase(0), "
            "Phase(0), "
            "Phase(0), "
            "Phase(0), "
            "Phase(1)"
          "], "
          "challenge_phase: ["
            "Phase(0), "
            "Phase(0)"
          "], "
          "gates: ["
            "Product("
              "Product("
                "Fixed { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}, "
                "Sum("
                  "Constant(0x0000000000000000000000000000000000000000000000000000000000000002), "
                  "Negated(Fixed { "
                    "query_index: 1, "
                    "column_index: 1, "
                    "rotation: Rotation(0) "
                  "})"
                ")"
              "), "
              "Sum("
                "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                "Negated(Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "})"
              ")"
            "), "
            "Product("
              "Product("
                "Fixed { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}, "
                "Sum("
                  "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                  "Negated(Fixed { "
                    "query_index: 1, "
                    "column_index: 1, "
                    "rotation: Rotation(0) "
                  "})"
                ")"
              "), "
              "Sum("
                "Constant(0x0000000000000000000000000000000000000000000000000000000000000001), "
                "Negated(Advice { "
                  "query_index: 0, "
                  "column_index: 4, "
                  "rotation: Rotation(0), "
                  "phase: Phase(1) "
                "})"
              ")"
            "), "
            "Product("
              "Fixed { "
                "query_index: 0, "
                "column_index: 0, "
                "rotation: Rotation(0) "
              "}, "
              "Sum("
                "Product("
                  "Advice { "
                    "query_index: 0, "
                    "column_index: 4, "
                    "rotation: Rotation(0), "
                    "phase: Phase(1) "
                  "}, "
                  "Sum("
                    "Sum("
                      "Product("
                        "Advice { "
                          "query_index: 1, "
                          "column_index: 0, "
                          "rotation: Rotation(0) "
                        "}, "
                        "Challenge(Challenge { "
                          "index: 0, "
                          "phase: Phase(0) "
                        "})"
                      "), "
                      "Advice { "
                        "query_index: 2, "
                        "column_index: 1, "
                        "rotation: Rotation(0) "
                      "}"
                    "), "
                    "Challenge(Challenge { "
                      "index: 1, "
                      "phase: Phase(0) "
                    "})"
                  ")"
                "), "
                "Negated(Product("
                  "Advice { "
                    "query_index: 5, "
                    "column_index: 4, "
                    "rotation: Rotation(1), "
                    "phase: Phase(1) "
                  "}, "
                  "Sum("
                    "Sum("
                      "Product("
                        "Advice { "
                          "query_index: 3, "
                          "column_index: 2, "
                          "rotation: Rotation(0) "
                        "}, "
                        "Challenge(Challenge { "
                          "index: 0, "
                          "phase: Phase(0) "
                        "})"
                      "), "
                      "Advice { "
                        "query_index: 4, "
                        "column_index: 3, "
                        "rotation: Rotation(0) "
                      "}"
                    "), "
                    "Challenge(Challenge { "
                      "index: 1, "
                      "phase: Phase(0) "
                    "})"
                  ")"
                "))"
              ")"
            ")"
          "], "
          "advice_queries: ["
            "("
              "Column { "
                "index: 4, "
                "column_type: Advice { "
                  "phase: Phase(1) "
                "} "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 1, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 2, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 3, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 4, "
                "column_type: Advice { "
                  "phase: Phase(1) "
                "} "
              "}, "
              "Rotation(1)"
            ")"
          "], "
          "instance_queries: [], "
          "fixed_queries: ["
            "("
              "Column { "
                "index: 0, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            "), "
            "("
              "Column { "
                "index: 1, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            ")"
          "], "
          "permutation: Argument { "
            "columns: [] "
          "}, "
          "lookups: [], "
          "constants: [], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x297ff8a661d1fa1196c065b6fb7df901fb16b5b83168ddca6c7749d963cc967a, "
            "0x1a7b1f2b5f3e35fc4c706ece6b8f646e95904f9aa417f8a31fd37a41da167ec1), "
          "(0x1025d577f7527c7c9a5d132164beef9ec3ebc63805b146843466f6cdf5fb37d4, "
            "0x1ba406f864ec730f7f725b3478ce1e22c5579b6228593c9ba36e9cb18a10aab3)"
        "], "
        "permutation: VerifyingKey { "
          "commitments: [] "
        "} "
      "}";
  // clang-format on

  constexpr static uint8_t kProof[] = {
      122, 117, 229, 163, 114, 52,  159, 223, 87,  191, 210, 4,   56,  250,
      59,  18,  206, 131, 78,  129, 60,  202, 130, 174, 203, 80,  0,   221,
      190, 150, 199, 170, 97,  250, 239, 195, 165, 188, 251, 230, 129, 105,
      212, 209, 41,  124, 75,  202, 166, 142, 57,  16,  160, 164, 175, 119,
      80,  96,  210, 1,   220, 55,  229, 150, 176, 0,   127, 136, 63,  27,
      146, 170, 55,  152, 223, 250, 131, 165, 92,  170, 126, 214, 34,  187,
      172, 236, 248, 250, 195, 21,  121, 118, 69,  108, 167, 132, 242, 42,
      222, 129, 23,  144, 206, 50,  2,   128, 3,   44,  222, 217, 223, 35,
      228, 184, 246, 76,  34,  146, 170, 5,   127, 243, 140, 134, 59,  224,
      63,  139, 97,  91,  176, 229, 208, 22,  81,  76,  102, 212, 230, 42,
      248, 115, 0,   57,  161, 155, 76,  46,  222, 63,  50,  191, 169, 117,
      109, 77,  100, 113, 115, 163, 219, 109, 33,  128, 52,  174, 106, 188,
      197, 159, 82,  64,  13,  77,  214, 88,  94,  3,   59,  239, 242, 15,
      57,  52,  184, 96,  8,   195, 219, 8,   28,  165, 23,  155, 225, 52,
      2,   100, 158, 139, 3,   255, 216, 141, 144, 92,  148, 43,  245, 117,
      105, 244, 203, 45,  169, 199, 189, 108, 165, 254, 123, 192, 88,  166,
      183, 219, 170, 20,  145, 67,  54,  188, 84,  208, 144, 229, 73,  102,
      239, 199, 192, 67,  84,  125, 77,  150, 208, 152, 34,  245, 28,  245,
      53,  241, 240, 142, 238, 116, 44,  171, 9,   62,  209, 106, 14,  52,
      209, 60,  194, 175, 33,  157, 149, 131, 25,  211, 81,  22,  167, 97,
      84,  92,  7,   120, 171, 195, 156, 29,  78,  254, 1,   92,  9,   249,
      182, 255, 222, 133, 161, 215, 117, 56,  201, 12,  65,  236, 138, 136,
      207, 214, 29,  1,   218, 176, 164, 120, 50,  120, 130, 42,  1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   96,  134, 180, 185, 62,  13,  187, 73,  198, 194, 240, 58,
      143, 101, 182, 194, 201, 129, 242, 182, 19,  7,   12,  75,  34,  92,
      65,  156, 85,  241, 183, 171, 45,  201, 146, 14,  118, 58,  117, 27,
      61,  108, 249, 137, 144, 71,  115, 19,  73,  98,  4,   77,  237, 52,
      69,  171, 85,  201, 246, 215, 91,  71,  145, 175, 139, 126, 48,  67,
      66,  130, 140, 51,  253, 115, 220, 250, 206, 22,  138, 113, 63,  0,
      74,  233, 167, 144, 210, 243, 212, 44,  42,  58,  221, 5,   215, 30,
      94,  146, 3,   134, 1,   201, 167, 90,  132, 197, 56,  187, 221, 23,
      242, 164, 87,  157, 142, 157, 80,  235, 12,  136, 21,  115, 181, 19,
      96,  199, 209, 5,   135, 209, 93,  98,  16,  78,  106, 217, 160, 233,
      35,  255, 207, 251, 154, 71,  20,  253, 239, 202, 139, 237, 246, 171,
      137, 255, 94,  92,  53,  190, 236, 34,  153, 4,   94,  63,  214, 40,
      181, 15,  147, 150, 81,  110, 181, 99,  38,  34,  56,  239, 28,  121,
      151, 11,  251, 82,  30,  105, 132, 33,  69,  126, 142, 15,  157, 153,
      218, 141, 253, 210, 254, 249, 230, 130, 27,  49,  161, 88,  161, 222,
      147, 49,  245, 92,  45,  16,  131, 95,  217, 194, 215, 16,  69,  71,
      78,  29,  108, 202, 1,   68,  66,  44,  39,  172, 207, 144, 77,  121,
      120, 114, 211, 147, 52,  200, 152, 197, 161, 235, 76,  139, 115, 141,
      206, 208, 134, 251, 78,  15,  16,  93,  166, 218, 134, 85,  60,  230,
      36,  206, 42,  75,  206, 2,   145, 43,  18,  145, 135, 107, 152, 229,
      5,   62,  206, 252, 129, 232, 114, 75,  228, 33,  140, 218, 27,  139,
      246, 80,  127, 36,  80,  115, 177, 219, 69,  80,  235, 123, 191, 251,
      189, 206, 103, 248, 242, 58,  173, 185, 101, 112, 138, 176, 235, 34,
      81,  60,  31,  132, 241, 220, 231, 212, 135, 74,  159, 124, 115, 227,
      149, 192, 56,  152, 43,  182, 158, 246, 63,  125, 148, 144, 243, 217,
      66,  119, 14,  38,  238, 129, 238, 25,  126, 177, 4,   135, 84,  255,
      216, 141, 221, 137, 172, 22,  182, 33,  217, 8,   95,  92,  208, 83,
      226, 5,   67,  65,  218, 137, 253, 12,  231, 216, 174, 122, 162, 239,
      88,  16,  65,  125, 101, 123, 96,  188, 118, 164, 68,  145, 203, 129,
      54,  95,  84,  4,   83,  98,  53,  87,  145, 106, 223, 26,  146, 169,
      158, 159, 209, 157, 89,  102, 204, 254, 97,  92,  149, 104, 21,  94,
      238, 231, 74,  192, 45,  11,  167, 112, 183, 54,  189, 61,  176, 204,
      151, 2,   216, 147, 249, 148, 195, 213, 197, 49,  184, 7,   174, 45,
      175, 251, 144, 126, 180, 32,  145, 186, 94,  84,  170, 71,  228, 213,
      109, 89,  141, 161, 164, 38,  204, 163, 86,  138, 200, 222, 184, 232,
      163, 254, 172, 253, 164, 60,  8,   91,  31,  113, 46,  55,  14,  133,
      120, 246, 172, 208, 57,  35,  123, 179, 212, 32,  1,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      209, 212, 200, 125, 223, 94,  117, 108, 135, 235, 223, 52,  22,  57,
      192, 220, 171, 220, 21,  114, 181, 209, 172, 136, 84,  166, 153, 21,
      191, 31,  241, 162, 67,  132, 199, 67,  95,  218, 218, 119, 67,  127,
      7,   147, 173, 104, 127, 197, 127, 234, 110, 49,  115, 143, 218, 23,
      132, 5,   35,  0,   217, 55,  177, 128};

  static void TestConfig(ShuffleCircuitConfig<F, kW>& config) {
    EXPECT_EQ(config.q_shuffle(), Selector::Simple(0));
    EXPECT_EQ(config.q_first(), Selector::Simple(1));
    EXPECT_EQ(config.q_last(), Selector::Simple(2));
    for (size_t i = 0; i < kW; i++) {
      EXPECT_EQ(config.original_column_keys()[i], AdviceColumnKey(i));
      EXPECT_EQ(config.shuffled_column_keys()[i], AdviceColumnKey(kW + i));
    }
    EXPECT_EQ(config.theta(), Challenge(0, kFirstPhase));
    EXPECT_EQ(config.gamma(), Challenge(1, kFirstPhase));
    EXPECT_EQ(config.z(), AdviceColumnKey(2 * kW, kSecondPhase));
  }

  static Circuit GetCircuit(size_t i = 0) {
    CHECK_LT(i, std::size(kOriginalTables));
    return {CreateTable(kOriginalTables[i]), CreateTable(kShuffledTables[i])};
  }

  static std::vector<Circuit> Get2Circuits() {
    return {GetCircuit(0), GetCircuit(1)};
  }

 private:
  static std::vector<std::vector<F>> CreateTable(
      const std::string_view table[kW][kH]) {
    return base::CreateVector(kW, [table](size_t i) {
      return base::CreateVector(
          kH, [table, i](size_t j) { return *F::FromHexString(table[i][j]); });
    });
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_CIRCUIT_TEST_DATA_H_

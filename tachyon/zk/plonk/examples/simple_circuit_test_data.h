#ifndef TACHYON_ZK_PLONK_EXAMPLES_SIMPLE_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_SIMPLE_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

#include <string_view>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/examples/circuit_test_data.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PCS, typename LS, typename SFINAE = void>
class SimpleTestData : public CircuitTestData<Circuit, PCS, LS> {};

// FloorPlanner = SimpleFloorPlanner
template <typename Circuit, typename PCS, typename LS>
class SimpleTestData<Circuit, PCS, LS,
                     std::enable_if_t<IsSimpleFloorPlanner<Circuit>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;

  // Set flags of values to be used as true
  constexpr static bool kAssemblyFixedColumnsFlag = true;
  constexpr static bool kAssemblyPermutationColumnsFlag = true;
  constexpr static bool kCycleStoreMappingFlag = true;
  constexpr static bool kCycleStoreAuxFlag = true;
  constexpr static bool kCycleStoreSizesFlag = true;
  constexpr static bool kLFirstFlag = true;
  constexpr static bool kLLastFlag = true;
  constexpr static bool kLActiveRowFlag = true;
  constexpr static bool kFixedColumnsFlag = true;
  constexpr static bool kFixedPolysFlag = true;
  constexpr static bool kPermutationsColumnsFlag = true;
  constexpr static bool kPermutationsPolysFlag = true;
  constexpr static bool kAdviceCommitmentsFlag = true;
  constexpr static bool kPermutationProductCommitmentsFlag = true;
  constexpr static bool kVanishingHPolyCommitmentsFlag = true;
  constexpr static bool kAdviceEvalsFlag = true;
  constexpr static bool kFixedEvalsFlag = true;
  constexpr static bool kCommonPermutationEvalsFlag = true;
  constexpr static bool kPermutationProductEvalsFlag = true;
  constexpr static bool kPermutationProductNextEvalsFlag = true;
  constexpr static bool kPermutationProductLastEvalsFlag = true;

  constexpr static size_t kN = 16;

  // clang-format off
  constexpr static std::string_view kPinnedConstraintSystem =
      "PinnedConstraintSystem { "
        "num_fixed_columns: 1, "
        "num_advice_columns: 2, "
        "num_instance_columns: 1, "
        "num_selectors: 1, "
        "gates: [Product("
          "Selector(Selector(0, true)), "
          "Sum("
            "Product("
              "Advice { "
                "query_index: 0, "
                "column_index: 0, "
                "rotation: Rotation(0) "
              "}, "
              "Advice { "
                "query_index: 1, "
                "column_index: 1, "
                "rotation: Rotation(0) "
              "}"
            "), "
            "Negated(Advice { "
              "query_index: 2, "
              "column_index: 0, "
              "rotation: Rotation(1) "
            "})"
          ")"
        ")], "
        "advice_queries: ["
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
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(1)"
          ")"
        "], "
        "instance_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Rotation(0)"
        ")], "
        "fixed_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Fixed "
          "}, "
          "Rotation(0)"
        ")], "
        "permutation: Argument { columns: ["
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Fixed "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 1, "
            "column_type: Advice "
          "}"
        "] }, "
        "lookups: [], "
        "constants: [Column { "
          "index: 0, "
          "column_type: Fixed "
        "}], "
        "minimum_degree: None "
      "}";
  // clang-format on

  constexpr static std::string_view kAssemblyFixedColumns[][kN] = {
      {
          "0x0000000000000000000000000000000000000000000000000000000000000007",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
  };

  constexpr static AnyColumnKey kAssemblyPermutationColumns[] = {
      InstanceColumnKey(0),
      FixedColumnKey(0),
      AdviceColumnKey(0),
      AdviceColumnKey(1),
  };

  // clang-format off
  constexpr static Label kCycleStoreMapping[][kN] = {
      {{2, 8}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 2}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 3}, {3, 3},  {2, 7},  {2, 0},  {3, 5},  {2, 4},  {3, 7},  {1, 0},
       {0, 0}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {3, 2},  {2, 1},  {3, 4},  {2, 5},  {3, 6},  {2, 6},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };

  constexpr static Label kCycleStoreAux[][kN] = {
      {{2, 8}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{1, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 3}, {3, 3},  {1, 0},  {2, 3},  {2, 5},  {2, 5},  {3, 7},  {1, 0},
       {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {3, 2},  {3, 3},  {3, 4},  {2, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };
  // clang-format on

  constexpr static size_t kCycleStoreSizes[][kN] = {
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1},
  };

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {false, false, false,  true, false,  true, false,  true,
       false, false, false, false, false, false, false, false}
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
          "num_advice_columns: 2, "
          "num_instance_columns: 1, "
          "num_selectors: 1, "
          "gates: [Product("
            "Fixed { "
              "query_index: 1, "
              "column_index: 1, "
              "rotation: Rotation(0) "
            "}, "
            "Sum("
              "Product("
                "Advice { "
                  "query_index: 0, "
                  "column_index: 0, "
                  "rotation: Rotation(0) "
                "}, "
                "Advice { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}"
              "), "
              "Negated(Advice { "
                "query_index: 2, "
                "column_index: 0, "
                "rotation: Rotation(1) "
              "})"
            ")"
          ")], "
          "advice_queries: ["
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
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(1)"
            ")"
          "], "
          "instance_queries: [("
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}, "
            "Rotation(0)"
          ")], "
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
          "permutation: Argument { columns: ["
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Fixed "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}"
          "] }, "
          "lookups: [], "
          "constants: ["
            "Column { "
              "index: 0, "
              "column_type: Fixed "
            "}"
          "], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x0f9cc629d0010671d7f755267acccfc8d5854f47d4e437ec84bddcc27b9f19d1, "
            "0x199b1dcc7aff518a4e49c6393bcf847cc3fde52ae69e326dea0d1d901424552a), "
          "(0x10472018b5bfdcc76f3925ea4f660dc7167ef12c96fb70f686d0c1cf7791cde5, "
            "0x0358f44f7cb29a8d129dbe6b61fc1d921a903f1e9209abc137c7a6446c3ae38f)"
        "], "
        "permutation: VerifyingKey { commitments: ["
          "(0x0365b8986f1c38476aa6479eea1b688244b4070413eb393efa5b06441ac2aeaa, "
            "0x303ed0aaed99cb2848d844239ab4ab9a9191b86544edab860c42a2d4e504cf34), "
          "(0x1af13f7dc79a97c1a690215e2ffd3d8386f682fe1a13bdb8588d2fbaa0161edc, "
            "0x10f0c0ddf9db782e88deb3e49464292b1276b2b9ad29b5a25971c85fe0b8aa96), "
          "(0x09837455c613e5b0e0edd2a1d47f1efdee21eeeda00548f3686c733301e23da4, "
            "0x2e2b7776741eb2214916eb80ae0982c32b02e57e83c51e8907b5ffeab048bf2d), "
          "(0x23e033260e2f2ed7dd27295a06d951fad7c1545a493f24e8d7f416ccc3c74670, "
            "0x0957c9d3b0c00ff783bc3357dd0b5cf891f49bc78f569126ba6e68bc394859ef)"
        "] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x03b30e0717f2047e825763ccf9c91fff91c82eef5ec0834f66f359f29a3d3b58";

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
          "0x0000000000000000000000000000000000000000000000000000000000000007",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
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
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
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
  };

  constexpr static std::string_view kFixedPolys[][kN] = {
      {
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
      },
      {
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x2e2956f8332a2abea8eae65e83211a8cfd4c9c38c6ed5c09186cafec19009edf",
          "0x2014447de15a99b6df03833e95f96ae1299c9ec6ff990b6e75fa3b3b04846a57",
          "0x130034a5a3705c18e67b698f576257c783c3403058f57e9a359495efd00ce8cf",
          "0x2144f5eefad21e1ca37ae273a4ee5b4a7eef137cbbeb1d198c9e59c37ef70364",
          "0x11ded077258dd59f58ab8525c92955ebcb2b1905e676b2fe47ca20d6919e5e16",
          "0x02b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e8",
          "0x152fae7ca9f4520815c46c7ae6996e0cd78f9e211ed4ffa8d8d48d83b3a445cd",
          "0x0912ceb58a394e07d28f0d12384840917789bb8d96d2c51b3cba5e0bbd000000",
          "0x023af77aae07756b0f655f57fe603dd02ae74c0fb2cc14882b7545a7d6ff6122",
          "0x105009f4ffd70672d94cc277eb87ed7bfe9749817a206522cde7ba58eb7b95aa",
          "0x1d6419cd3dc14410d1d4dc272a1f0095a470a81820c3f1f70e4d5fa41ff31732",
          "0x0f1f5883e65f820d14d56342dc92fd12a944d4cbbdce5377b7439bd07108fc9d",
          "0x1e857dfbbba3ca8a5fa4c090b85802715d08cf429342bd92fc17d4bd5e61a1eb",
          "0x2db11694c4a58b3789868bd388165969c30dc1120a37fff09a991abf43e42a19",
          "0x1b349ff6373d4e21a28bd93b9ae7ea5050a44a275ae470e86b0d68103c5bba34",
      },
  };

  constexpr static std::string_view kPermutationsColumns[][kN] = {
      {
          "0x1cb0ed9df901b713b97ee5357df1bf9b16f16cc0d737b31e07ba77d910efc8d6",
          "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
          "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
          "0x107aab49e65a67f9da9cd2abf78be38bd9dc1d5db39f81de36bcfa5b4b039043",
          "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
          "0x2290ee31c482cf92b79b1944db1c0147635e9004db8c3b9d13644bef31ec3bd3",
          "0x1d59376149b959ccbd157ac850893a6f07c2d99b3852513ab8d01be8e846a566",
          "0x2d8040c3a09c49698c53bfcb514d55a5b39e9b17cb093d128b8783adb8cbd723",
          "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
          "0x0f5c21d0ca65e0db9be1f670eca407d08ec5ec67626a74f892ccebcd0cf9b9f6",
          "0x0530d09118705106cbb4a786ead16926d5d174e181a26686af5448492e42a181",
          "0x1fe9a328fad7382fddb3730a89f574d14e57caeac619eeb30d24fb38a4fc6fbe",
          "0x0000000000000000b3c4d79d41a91758cb49c3517c4604a520cff123608fc9cb",
          "0x0dd360411caed09700b52c71a6655715c4d558439e2d34f4307da9a4be13c42e",
          "0x130b17119778465cfb3acaee30f81dee20710ead41671f568b11d9ab07b95a9b",
          "0x02e40daf409556c02bfc85eb303402b774954d30aeb0337eb85a71e6373428de",
      },
      {
          "0x0b6f861977ce57ddb2e647048ed9b9433cac640ba0599e264e24446765d915d3",
          "0x133f51f46c2a51201fbb89dc1d6868acfef0f7ce0e572ccb98e04e99eac3ea36",
          "0x1240a374ee6f71e12df10a129946ca9ba824f58a50e3a58b4c68dd5590b74ad8",
          "0x009553089042fe83ab570b54e4e64d307d8e8b20d568e782ece3926981d0a96c",
          "0x14a6c152ace4b16a42e1377e400798e15ac60320c21d90277890dbdd551d1912",
          "0x035992598be4d2ae5334f24f30355a1fca7f0e28762a178d79e993dca70eaabf",
          "0x03b645319eb70d7a692ea8c87fbcab1f292cd502071d9ffb7291ebe28356aa07",
          "0x231e38741f5c3f0ce2509bd3289b09a893b14b136eea6ed00cec9ac9271c9db5",
          "0x2741e304be6aaf5f53641f0bacb8e9ebccd45eba1b23316bbcd39ed80acc165f",
          "0x1d24fc7e75074f099894bbda6418efb02942f07a6b6243c5ab01a6fa053c15cb",
          "0x1e23aafdf2c22e488a5f3ba3e83a8dc1800ef2be28d5cb05f779183e5f48b529",
          "0x2fcefb6a50eea1a60cf93a619c9b0b2caaa55d27a450890e56fe632a6e2f5695",
          "0x1bbd8d20344ceebf756f0e384179bf7bcd6de527b79be069cb5119b69ae2e6ef",
          "0x2d0abc19554ccd7b651b5367514bfe3d5db4da20038f5903c9f861b748f15542",
          "0x2cae0941427a92af4f219cee01c4ad3dff071346729bd095d15009b16ca955fa",
          "0x0d4615fec1d5611cd5ffa9e358e64eb494829d350acf01c136f55acac8e3624c",
      },
      {
          "0x26f93d99832c6285d9abf8c5b896ea753ed137516972c5ddcc464b56c488d600",
          "0x0f34fda7cd268bf095354ea6c2631826c349d7518bf094361bc91f61c519c7fb",
          "0x2a3d1fef5cb3ce1065d222dde5f9b16d48b42597868c1a49e15cb8007c8778a4",
          "0x13b360d4e82fe915fed16081038f98c211427b87a281bd733c277dbadf10372b",
          "0x1e626bf9ef3c8920522383ff9be21287c3af8d47fe61ff9af6c6d8d0118154bc",
          "0x086398ace043cf0db4e99f2712f392ddbd9c7b9202b2fcadf26e9b145316df9f",
          "0x2a150ee7450ad90cace368ea424c8fdf36b3821e79f5f1c617a5f8e74c19b4cf",
          "0x09226b6e22c6f0ca64ec26aad4c86e715b5f898e5e963f25870e56bbe533e9a2",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x277c827440297ddeb0d85560fc7da91bcfd8729cec6bf01c3ecc664827388e23",
          "0x24f4c8596963484c0569feb1f2a79f19eb87843cd95fd26af5bdb12c8a26ea2e",
          "0x096b10d95e053da3dea44cf0c8ea6de7e962b0f71046aab3779baa3d2b772a01",
          "0x2800b5c600edd11c0366a68f6e8dc57f6a976cb6770673e351735a7f9ce92062",
          "0x2bedf321de5b3dacbb8cbc7312ea9c937dec5a7d07ae1c24ccdbc51c4bf6ef33",
          "0x022a8cf5a31c990f250e75e7df4daafe02929a5e514802a8b8fe8be7c366bb55",
          "0x06272e83847dd219527e22d89b87a6efdf7fc2b0f32d564762853d937378875d",
      },
      {
          "0x18afdf23e9bd9302673fc1e076a492d4d65bd18ebc4d854ed189139bab313e52",
          "0x2f0e061e83e8b12c1bdf8df7cca02483295e89f78a462ce38859661e06501815",
          "0x034183d253b6b250dae0a457797029523434d2b6bc41c09b6ef409bd970e4208",
          "0x08e7cbfea108224b0777f0558503af41585b75ab8d4d807505158f4bc8c771de",
          "0x2f549305063b1803a77ba92486c5383e00468fc857ec66d944868c4a309e596a",
          "0x04765b5102d6627cfcc389436e96bbc9aa478dcb720b546c77063077a40910ce",
          "0x02898db49f7022cac0aeabf721a3c7ab96e14ad35e7d6b0410b868ff504b62fc",
          "0x2e39c17d3e15071a9341cfcea233ad5f25a14dea28716de88ae369ac2c9944ac",
          "0x17b46f4ef7740d27511083d60adcc58851d816b9bd6beb427258e1f844cec1af",
          "0x015648545d48eefd9c70b7beb4e133d9fed55e50ef7343adbb888f75e9afe7ec",
          "0x2d22caa08d7aedd8dd6fa15f08112f0af3ff1591bd77aff5d4edebd658f1bdf9",
          "0x212f50cb140b1439231af70fbf1e403664ea10f6edc8dc5b2818d6322ae63806",
          "0x010fbb6ddaf6882610d49c91fabc201f27ed588021cd09b7ff5b6949bf61a697",
          "0x1201e278f1f51709662cc1b6e59f45d564845b007b5770f64d1b1cc3de7eab45",
          "0x2ddac0be41c17d5ef7a199bf5fdd90b191529d751b3c058d33298c949fb49d05",
          "0x064f3f8b9c26c71d0b6cdccc3f34c87df1806629ffc37ecb2c3bfcaca3e64b32",
      },
  };

  constexpr static std::string_view kPermutationsPolys[][kN] = {
      {
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8f",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
          "0x231004c8da62398dea4f1e40d0e808b9bd12c67de122f895bf270053460efc8e",
      },
      {
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x0640f831aa044d38fe46c45508516d98a6f118b1ab16de0c7f41963d5e3e3c65",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
          "0x2d82db36686efc9851aae360b50a578473c5776bc63a0f783c153515690a52c4",
      },
      {
          "0x18038fb09dafe4561b7c81a4ddecb5a9fdec9905c914eb53873760ec477b6153",
          "0x01330c7cbea5f36f17d3698ffdacd0e8d561fc0253004050c673cb4825a5d678",
          "0x11bd06280c7c8a6daa71eb9d3b2c97f9f211a37a41798bc63f78d5af70135d20",
          "0x021ebc7877f2b98917ec49b986e1a497743d0464e2e1cf787fcc843b013626bb",
          "0x2965a7e65ce635bbab94a2fffae4c86626d67c0eb10d658380f782f08db31ef9",
          "0x13d9914eeb3084263023150484b8da37654dc6d57f82afffca2739b48de541c3",
          "0x150b22b022aca6d5eacd65a0435fe2b487242afc5df5b9a417a001d5a79554a0",
          "0x1911d0efc0d670781ad228d1df69e12aa502675c2c2b0873421dc9f32f16364f",
          "0x173bca2a5af56060d72e1cf66a4c15350c7a1d0985d14232e9c548bf83a8e270",
          "0x1761ac780a2688e48980ee5f7bb6c6489891ad7207f1b53c9a9aecdf3ca68117",
          "0x2af4468570b8b17c26c074b53131f4dab8f4abc05ac204dfee977cd8f2ec9c0e",
          "0x29a949ffe203c1114c6cc31a6b99269932f7cd7340abbe797c0d6c99a7f802f1",
          "0x2f93c69781f9dd1bcc42e4149107242b847da4d909a47893ebe85b9a1fe5700e",
          "0x2a7dc0ef215246ec520423ffc5f6e6fa3d5c0144e341d3358dd64ce8e648afff",
          "0x134ac81b3f8e47b083bd4a759f2e5be3ec6cdfc0d4b28794740f1a3096f24a6d",
          "0x2115cec3e5a7af5c4f4829c80b9825c24f499de351535dc35cb607a47b85c1b7",
      },
      {
          "0x28967ba8c313c600d17a4ddd20a4f32f49d6e1e9fcba5e5249324a66d96b4f5e",
          "0x00a354bfcc60f6a39aa68e9a9fa36f4f2bc9322db22d4c76b20738f12376fa40",
          "0x2d0e02c7383ab7a8d1dc324434cbc0efa7c7da0a5c10f1008c072ad9284c951b",
          "0x295fc9ac7c7023fa3374d3df0959578ede07bc8e8e32a7d46b5928ac0f6051c7",
          "0x2be457106435d313da2cbe5c1304c9b5eac6cef2e824ab55be7e094f779a7680",
          "0x11d0a6eb25ba5944c820e99c5b6bfeacbaed201b58985a3e47ac129d746acb2d",
          "0x10f99e36b6e6090fddea8d5f4048b35d3bf6c5f9b4b84f83c3f8695c1fc00a84",
          "0x1269a9b9c8dd3f9f671c12f107a14cb0be65a15412dc8d8b0fb139184bfbd050",
          "0x07cdd2ca1e1dda28e6d5f7d960dc652dde5d065e7cff123efaafab2d1694b0a3",
          "0x180c8a641d5c9c5ecc993345d7012385aa929f610a2038d81f81daaa87ba4412",
          "0x03564baba8f6e880e67413724cb5976d806c0e3e1da87f90b7dacabac7b36ae6",
          "0x070484c664c17c2f84db71d7782800ce4a2c2bb9eb86c8bcd888cce7e09fae3a",
          "0x047ff7627cfbcd15de23875a6e7c8ea73d6d19559194c53b8563ec4478658981",
          "0x1e93a787bb7746e4f02f5c1a261559b06d46c82d21211652fc35e2f67b9534d4",
          "0x1f6ab03c2a4b9719da65b8574138a4ffec3d224ec501210d7fe98c37d03ff57d",
          "0x1dfaa4b91854608a513432c579e00bac69ce46f466dce3063430bc7ba4042fb1",
      },
  };

  constexpr static uint8_t kProof[] = {
      206, 109, 139, 136, 181, 35,  204, 231, 212, 93,  105, 116, 154, 77,  204,
      23,  71,  148, 11,  151, 126, 145, 6,   150, 171, 185, 254, 230, 41,  136,
      76,  141, 132, 227, 154, 206, 134, 35,  253, 67,  8,   186, 228, 143, 116,
      139, 145, 119, 85,  253, 127, 208, 95,  153, 195, 112, 209, 116, 172, 45,
      15,  175, 128, 142, 206, 109, 139, 136, 181, 35,  204, 231, 212, 93,  105,
      116, 154, 77,  204, 23,  71,  148, 11,  151, 126, 145, 6,   150, 171, 185,
      254, 230, 41,  136, 76,  141, 132, 227, 154, 206, 134, 35,  253, 67,  8,
      186, 228, 143, 116, 139, 145, 119, 85,  253, 127, 208, 95,  153, 195, 112,
      209, 116, 172, 45,  15,  175, 128, 142, 240, 229, 40,  5,   109, 36,  59,
      227, 58,  205, 89,  157, 199, 193, 252, 51,  168, 195, 186, 126, 190, 9,
      14,  29,  214, 95,  182, 184, 134, 94,  46,  41,  106, 156, 194, 192, 74,
      238, 43,  10,  112, 155, 176, 178, 14,  10,  189, 165, 211, 91,  90,  42,
      17,  165, 116, 23,  78,  117, 45,  125, 110, 1,   233, 170, 200, 135, 181,
      143, 22,  0,   52,  134, 185, 92,  30,  3,   65,  216, 74,  165, 18,  77,
      5,   55,  15,  109, 186, 248, 156, 119, 247, 24,  70,  41,  1,   16,  62,
      36,  174, 136, 24,  93,  149, 45,  191, 112, 228, 131, 82,  170, 188, 118,
      86,  196, 76,  197, 11,  167, 11,  216, 230, 14,  231, 164, 157, 39,  49,
      38,  251, 240, 47,  6,   163, 23,  217, 174, 233, 203, 174, 0,   205, 128,
      144, 251, 60,  132, 157, 32,  33,  112, 229, 156, 139, 47,  162, 5,   202,
      84,  199, 47,  27,  98,  96,  122, 158, 193, 213, 226, 200, 103, 5,   218,
      173, 125, 15,  210, 123, 98,  148, 53,  48,  206, 127, 172, 107, 33,  234,
      174, 142, 79,  186, 134, 142, 249, 224, 82,  159, 246, 235, 120, 167, 5,
      48,  155, 25,  120, 55,  94,  246, 75,  242, 253, 253, 152, 55,  94,  67,
      65,  174, 82,  157, 209, 177, 132, 71,  170, 194, 199, 129, 201, 108, 111,
      231, 243, 252, 145, 116, 7,   88,  87,  28,  81,  97,  30,  116, 31,  167,
      99,  44,  21,  104, 160, 51,  70,  32,  5,   1,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   159, 45,  146, 241,
      184, 125, 165, 48,  148, 73,  242, 94,  170, 139, 162, 0,   21,  231, 23,
      74,  94,  214, 195, 249, 139, 29,  249, 198, 235, 204, 224, 27,  135, 78,
      126, 51,  100, 157, 42,  110, 141, 157, 29,  152, 51,  209, 243, 76,  0,
      200, 96,  204, 134, 244, 148, 187, 87,  7,   206, 106, 36,  190, 225, 163,
      221, 191, 109, 9,   219, 247, 181, 207, 188, 191, 31,  65,  129, 49,  12,
      89,  46,  245, 184, 213, 17,  43,  232, 225, 2,   80,  183, 171, 82,  185,
      159, 35,  189, 66,  107, 20,  150, 2,   185, 176, 238, 76,  171, 30,  52,
      254, 113, 233, 150, 92,  5,   25,  77,  216, 65,  95,  79,  148, 13,  116,
      24,  183, 14,  46,  67,  229, 240, 109, 227, 68,  145, 172, 135, 53,  164,
      146, 172, 217, 97,  115, 45,  46,  135, 153, 141, 33,  237, 14,  126, 112,
      239, 91,  249, 224, 14,  16,  221, 191, 109, 9,   219, 247, 181, 207, 188,
      191, 31,  65,  129, 49,  12,  89,  46,  245, 184, 213, 17,  43,  232, 225,
      2,   80,  183, 171, 82,  185, 159, 35,  189, 66,  107, 20,  150, 2,   185,
      176, 238, 76,  171, 30,  52,  254, 113, 233, 150, 92,  5,   25,  77,  216,
      65,  95,  79,  148, 13,  116, 24,  183, 14,  46,  67,  229, 240, 109, 227,
      68,  145, 172, 135, 53,  164, 146, 172, 217, 97,  115, 45,  46,  135, 153,
      141, 33,  237, 14,  126, 112, 239, 91,  249, 224, 14,  16,  36,  203, 226,
      64,  99,  130, 12,  59,  222, 24,  94,  88,  253, 218, 172, 0,   25,  215,
      133, 149, 31,  158, 7,   250, 173, 225, 133, 200, 90,  69,  47,  25,  98,
      34,  19,  103, 109, 218, 148, 30,  225, 90,  104, 37,  58,  197, 245, 62,
      9,   238, 44,  199, 61,  72,  2,   28,  51,  231, 189, 28,  3,   29,  35,
      18,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   185, 220, 125, 19,  203, 252, 236, 223, 117, 57,  175, 183,
      40,  37,  79,  31,  117, 165, 118, 41,  185, 125, 170, 158, 157, 74,  166,
      57,  64,  149, 206, 40,  120, 244, 169, 170, 133, 173, 68,  101, 140, 48,
      109, 103, 247, 3,   28,  4,   36,  59,  132, 43,  93,  119, 152, 208, 233,
      222, 219, 94,  136, 106, 20,  11,  87,  104, 205, 179, 87,  159, 234, 122,
      152, 191, 255, 13,  15,  38,  155, 81,  173, 252, 150, 180, 173, 179, 196,
      214, 30,  37,  199, 22,  203, 126, 5,   43,  105, 245, 116, 211, 118, 74,
      43,  166, 161, 118, 110, 238, 30,  223, 80,  204, 218, 180, 236, 139, 207,
      173, 229, 182, 166, 225, 169, 205, 21,  196, 193, 24,  145, 35,  130, 126,
      116, 70,  55,  29,  82,  53,  95,  80,  69,  203, 254, 226, 83,  17,  151,
      238, 128, 120, 105, 242, 98,  98,  11,  81,  88,  20,  116, 14,  55,  241,
      196, 81,  126, 218, 207, 52,  163, 159, 251, 43,  165, 147, 33,  107, 18,
      209, 194, 240, 179, 217, 121, 221, 195, 5,   152, 146, 169, 81,  170, 34,
      32,  97,  114, 73,  185, 104, 109, 151, 124, 157, 118, 48,  123, 213, 18,
      111, 5,   102, 78,  130, 237, 77,  157, 116, 202, 124, 90,  51,  224, 114,
      159, 45,  197, 61,  251, 242, 253, 129, 64,  250, 54,  14,  49,  62,  37,
      97,  226, 172, 109, 168, 14,  33,  113, 47,  243, 44,  106, 229, 10,  62,
      87,  83,  43,  7,   92,  6,   195, 239, 233, 58,  92,  12,  233, 115, 233,
      81,  118, 184, 252, 142, 76,  197, 214, 144, 16,  95,  246, 169, 9,   78,
      237, 145, 235, 141, 136, 1,   165, 83,  192, 177, 140, 190, 28,  139, 172,
      249, 119, 129, 211, 92,  226, 51,  226, 185, 203, 21,  121, 174, 115, 152,
      110, 69,  200, 142, 162, 71,  220, 22,  186, 134, 221, 207, 39,  208, 161,
      130, 234, 61,  27,  72,  233, 172, 187, 198, 231, 11,  120, 243, 25,  89,
      77,  38,  86,  20,  89,  80,  237, 100, 142, 41,  136, 32,  138, 31,  215,
      84,  109, 62,  66,  227, 100, 175, 18,  174, 58,  112, 169, 193, 95,  240,
      222, 15,  61,  236, 91,  90,  188, 94,  150, 212, 237, 16,  117, 27,  177,
      202, 190, 133, 200, 85,  173, 174, 13,  89,  177, 186, 65,  146, 113, 41,
      206, 142, 237, 73,  51,  196, 214, 34,  224, 184, 100, 209, 13,  29,  179,
      244, 71,  38,  7,   16,  137, 20,  184, 72,  209, 27,  255, 185, 144, 75,
      35,  7,   36,  68,  119, 203, 89,  205, 4,   242, 26,  190, 212, 139, 57,
      26,  6,   252, 141, 116, 125, 123, 64,  138, 127, 97,  121, 221, 197, 244,
      112, 113, 155, 70,  96,  133, 82,  7,   149, 25,  118, 141, 20,  42,  46,
      88,  239, 42,  7,   96,  209, 165, 40,  191, 164, 67,  8,   89,  152, 7,
      200, 100, 248, 171, 41,  218, 69,  252, 118, 179, 67,  9,   182, 148, 121,
      104, 153, 44,  247, 9,   218, 204, 78,  16,  182, 29,  29,  0,   119, 160,
      190, 17,  211, 11,  23,  241, 100, 12,  180, 243, 34,  102, 12,  187, 41,
      193, 69,  50,  84,  21,  221, 23,  204, 183, 175, 109, 240, 93,  120, 136,
      54,  66,  184, 194, 85,  59,  133, 86,  220, 246, 193, 244, 7,   216, 135,
      59,  92,  158, 65,  144, 41,  220, 100, 28,  21,  111, 26,  217, 23,  179,
      65,  205, 54,  26,  240, 52,  150, 157, 87,  210, 76,  217, 152, 132, 71,
      150, 10,  95,  77,  14,  228, 86,  178, 21,  103, 6,   74,  89,  231, 106,
      134, 147, 137, 174, 39,  174, 48,  221, 254, 215, 4,   39,  230, 15,  196,
      75,  24,  245, 239, 152, 208, 193, 187, 175, 132, 241, 195, 34,  38,  55,
      57,  17,  54,  64,  27,  138, 144, 11,  157, 37,  36,  5,   139, 117, 212,
      173, 39,  85,  164, 25,  206, 217, 149, 150, 4,   250, 77,  130, 115, 9,
      143, 200, 219, 182, 151, 215, 233, 194, 240, 248, 177, 123, 131, 209, 152,
      81,  102, 187, 53,  150, 25,  204, 189, 65,  9,   86,  79,  99,  81,  250,
      151, 4,   11,  172, 72,  139, 204, 43,  44,  115, 202, 121, 48,  124, 236,
      53,  203, 245, 141, 110, 228, 145, 107, 49,  56,  58,  168, 222, 112, 147,
      197, 179, 83,  10,  128, 240, 51,  82,  135, 247, 196, 23,  27,  22,  177,
      117, 78,  249, 227, 93,  64,  8,   176, 132, 143, 218, 225, 185, 123, 159,
      78,  200, 197, 254, 171, 0,   103, 14,  109, 105, 100, 65,  235, 96,  115,
      178, 20,  89,  249, 152, 94,  113, 124, 134, 89,  134, 176, 129, 249, 140,
      162, 116, 135, 158, 230, 87,  9,   32,  227, 25,  158, 147, 29,  22,  77,
      43,  83,  154, 15,  112, 182, 222, 83,  80,  108, 218, 132, 95,  207, 89,
      134, 152, 228, 205, 152, 172, 107, 188, 119, 46,  154, 108, 59,  219, 34,
      8,   146, 12,  91,  13,  166, 147, 124, 190, 65,  228, 254, 170, 42,  228,
      225, 100, 186, 38,  12,  202, 16,  176, 133, 49,  60,  16,  237, 211, 195,
      59,  246, 51,  38,  12,  208, 190, 66,  35,  23,  195, 152, 31,  96,  15,
      198, 165, 206, 149, 89,  130, 172, 23,  119, 105, 140, 216, 51,  6};

  // clang-format off
  constexpr static Point kAdviceCommitments[][2] = {
      {
          {"0x0d4c8829e6feb9ab9606917e970b944717cc4d9a74695dd4e7cc23b5888b6dce",
            "0x03a99ef4660a95515763e072043119fcbf6d3f3b709af6bf05b5c8b4d815a775"},
          {"0x0e80af0f2dac74d170c3995fd07ffd5577918b748fe4ba0843fd2386ce9ae384",
            "0x058b31b773e7a0e22f1ef9d6bbcc154b3dfaec09ff6c78084c1ae5c150f6624d"},
      },
      {
          {"0x0d4c8829e6feb9ab9606917e970b944717cc4d9a74695dd4e7cc23b5888b6dce",
            "0x03a99ef4660a95515763e072043119fcbf6d3f3b709af6bf05b5c8b4d815a775"},
          {"0x0e80af0f2dac74d170c3995fd07ffd5577918b748fe4ba0843fd2386ce9ae384",
            "0x058b31b773e7a0e22f1ef9d6bbcc154b3dfaec09ff6c78084c1ae5c150f6624d"},
      },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x2b059aeea380dae6d29c1f1709cddfe7a7fa5d239d2ed38f7c5e7db17349d035";

  constexpr static std::string_view kBeta =
      "0x14b7ba74ee783acecf23166da4c9ca6f4fe63cdc66758cae396ce0804e87d509";

  constexpr static std::string_view kGamma =
      "0x12cbf1b371cc5262a7e4a2eda2e1c3171f84b76f64183d5c90305b8b65efe947";

  // clang-format off
  constexpr static Point kPermutationProductCommitments[][4] = {
      {
          {"0x292e5e86b8b65fd61d0e09be7ebac3a833fcc1c79d59cd3ae33b246d0528e5f0",
           "0x1f266354b24e20bbae2fa25ffe113392c1b5df9ec5f2a2eb99f6cb5447f4fc24"},
          {"0x2ae9016e7d2d754e1774a5112a5a5bd3a5bd0a0eb2b09b700a2bee4ac0c29c6a",
           "0x15c499ad168f1446aaa20c9af669210de2ad0321c0b338110cbcaf0b1a331023"},
          {"0x1001294618f7779cf8ba6d0f37054d12a54ad841031e5cb9863400168fb587c8",
           "0x19f8d635ee9d52a4ad56ea25cfedaccb9e644bc89ef114036c23c137a6d6dc7a"},
          {"0x2631279da4e70ee6d80ba70bc54cc45676bcaa5283e470bf2d955d1888ae243e",
           "0x2085ebbe76f114b6287f8db2fbadeb9c88a72a94e13c2a280fa4a1823693f47c"},
      },
      {
          {"0x2fc754ca05a22f8b9ce57021209d843cfb9080cd00aecbe9aed917a3062ff0fb",
           "0x148f23b9c4ee958cdc46ee3bb7cb8f40b00e1f05c6944fbb166c87d4a34a64bc"},
          {"0x06ba4f8eaeea216bac7fce303594627bd20f7dadda0567c8e2d5c19e7a60621b",
           "0x1b9af694f8f7657a82b27d7376f29e723215a607780b20d9a6064938e58a3203"},
          {"0x04b1d19d52ae41435e3798fdfdf24bf65e3778199b3005a778ebf69f52e0f98e",
           "0x09bedea0070365f54dc7edffce4ddc96e83c74eaa8543e069f8ad7a56f5ef209"},
          {"0x05204633a068152c63a71f741e61511c5758077491fcf3e76f6cc981c7c2aa47",
           "0x01e44060b9b70dc9474317dc324a516f94878ade0ad8221ab46c5b0d5607cad4"},
      },
  };
  // clang-format on

  constexpr static std::string_view kY =
      "0x2b635a1a9615e175bf98c02fcd0eca6a0196c48c2be713ad3c42089ede337d69";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x1be0ccebc6f91d8bf9c3d65e4a17e71500a28baa5ef2499430a57db8f1922d9f",
       "0x0d6ebddee89b00c919f56501344541d27387bd6523b31e4aab73b885c7c70c16"},
      {"0x23e1be246ace0757bb94f486cc60c8004cf3d133981d9d8d6e2a9d64337e4e87",
       "0x277166abda65f12d816f6294fff5f0a258fc37025859ce3ea21a82315aa0d0ef"},
  };

  constexpr static std::string_view kX =
      "0x1171a27273726e2363fc9167880d5cba29092819cb46c22594092c9a6bd3fc34";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x239fb952abb75002e1e82b11d5b8f52e590c3181411fbfbccfb5f7db096dbfdd",
          "0x2e0eb718740d944f5f41d84d19055c96e971fe341eab4ceeb0b90296146b42bd",
          "0x100ee0f95bef707e0eed218d99872e2d7361d9ac92a43587ac9144e36df0e543",
      },
      {
          "0x239fb952abb75002e1e82b11d5b8f52e590c3181411fbfbccfb5f7db096dbfdd",
          "0x2e0eb718740d944f5f41d84d19055c96e971fe341eab4ceeb0b90296146b42bd",
          "0x100ee0f95bef707e0eed218d99872e2d7361d9ac92a43587ac9144e36df0e543",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x192f455ac885e1adfa079e1f9585d71900acdafd585e18de3b0c826340e2cb24",
      "0x12231d031cbde7331c02483dc72cee093ef5c53a25685ae11e94da6d67132262",
  };

  constexpr static std::string_view kCommonPermutationEvals[] = {
      "0x28ce954039a64a9d9eaa7db92976a5751f4f2528b7af3975dfecfccb137ddcb9",
      "0x0b146a885edbdee9d098775d2b843b24041c03f7676d308c6544ad85aaa9f478",
      "0x2b057ecb16c7251ed6c4b3adb496fcad519b260f0dffbf987aea9f57b3cd6857",
      "0x18c1c415cda9e1a6b6e5adcf8becb4dacc50df1eee6e76a1a62b4a76d374f569",
  };

  constexpr static std::string_view kPermutationProductEvals[][4] = {
      {
          "0x0e741458510b6262f2697880ee971153e2fecb45505f35521d3746747e822391",
          "0x072b53573e0ae56a2cf32f71210ea86dace261253e310e36fa4081fdf2fb3dc5",
          "0x298e64ed50591456264d5919f3780be7c6bbace9481b3dea82a1d027cfdd86ba",
          "0x1a398bd4be1af204cd59cb77442407234b90b9ff1bd148b8148910072647f4b3",
      },
      {
          "0x09f72c99687994b60943b376fc45da29abf864c80798590843a4bf28a5d16007",
          "0x066715b256e40e4d5f0a96478498d94cd2579d9634f01a36cd41b317d91a6f15",
          "0x0497fa51634f560941bdcc199635bb665198d1837bb1f8f0c2e9d797b6dbc88f",
          "0x200957e69e8774a28cf981b08659867c715e98f95914b27360eb4164696d0e67",
      },
  };

  constexpr static std::string_view kPermutationProductNextEvals[][4] = {
      {
          "0x22aa51a9929805c3dd79d9b3f0c2d1126b2193a52bfb9fa334cfda7e51c4f137",
          "0x01888deb91ed4e09a9f65f1090d6c54c8efcb87651e973e90c5c3ae9efc3065c",
          "0x10edd4965ebc5a5bec3d0fdef05fc1a9703aae12af64e3423e6d54d71f8a2088",
          "0x2aef582e2a148d76199507528560469b7170f4c5dd79617f8a407b7d748dfc06",
      },
      {
          "0x17dd15543245c129bb0c6622f3b40c64f1170bd311bea077001d1db6104eccda",
          "0x22c3f184afbbc1d098eff5184bc40fe62704d7fedd30ae27ae8993866ae7594a",
          "0x0a53b3c59370dea83a38316b91e46e8df5cb35ec7c3079ca732c2bcc8b48ac0b",
          "0x2e77bc6bac98cde4988659cf5f84da6c5053deb6700f9a532b4d161d939e19e3",
      },
  };

  constexpr static std::string_view kPermutationProductLastEvals[][4] = {
      {
          "0x2d9f72e0335a7cca749d4ded824e66056f12d57b30769d7c976d68b949726120",
          "0x16dc47a28ec8456e9873ae7915cbb9e233e25cd38177f9ac8b1cbe8cb1c053a5",
          "0x1d0dd164b8e022d6c43349ed8ece29719241bab1590daead55c885becab11b75",
          "",
      },
      {
          "0x1c64dc2990419e5c3b87d807f4c1f6dc56853b55c2b8423688785df06dafb7cc",
          "0x0973824dfa049695d9ce19a45527add4758b0524259d0b908a1b403611393726",
          "0x00abfec5c84e9f7bb9e1da8f84b008405de3f94e75b1161b17c4f7875233f080",
          "",
      },
  };

  constexpr static std::string_view kHEval =
      "0x1535a994ce99cdd4b9d17531a2925be1decc650d939dc63753f4fce93d78a5e4";

  static void TestConfig(FieldConfig<F>& config) {
    std::array<AdviceColumnKey, 2> expected_advice = {
        AdviceColumnKey(0),
        AdviceColumnKey(1),
    };
    EXPECT_EQ(config.advice(), expected_advice);
    EXPECT_EQ(config.instance(), InstanceColumnKey(0));
    EXPECT_EQ(config.s_mul(), Selector::Simple(0));
  }

  static Circuit GetCircuit() {
    F constant(7);
    F a(2);
    F b(3);
    return Circuit(std::move(constant), std::move(a), std::move(b));
  }

  static std::vector<Circuit> Get2Circuits() {
    Circuit circuit = GetCircuit();
    return {circuit, std::move(circuit)};
  }

  static std::vector<Evals> GetInstanceColumns() {
    F constant(7);
    F a(2);
    F b(3);
    F c = std::move(constant) * std::move(a).Square() * std::move(b).Square();
    std::vector<F> instance_column = {std::move(c)};
    return {Evals(std::move(instance_column))};
  }
};

// FloorPlanner = V1FloorPlanner
template <typename Circuit, typename PCS, typename LS>
class SimpleTestData<Circuit, PCS, LS,
                     std::enable_if_t<IsV1FloorPlanner<Circuit>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;

  // Set flags of values to be used as true
  constexpr static bool kAssemblyFixedColumnsFlag = true;
  constexpr static bool kAssemblyPermutationColumnsFlag = true;
  constexpr static bool kCycleStoreMappingFlag = true;
  constexpr static bool kCycleStoreAuxFlag = true;
  constexpr static bool kCycleStoreSizesFlag = true;
  constexpr static bool kLFirstFlag = true;
  constexpr static bool kLLastFlag = true;
  constexpr static bool kLActiveRowFlag = true;
  constexpr static bool kFixedColumnsFlag = true;
  constexpr static bool kFixedPolysFlag = true;
  constexpr static bool kPermutationsColumnsFlag = true;
  constexpr static bool kPermutationsPolysFlag = true;
  constexpr static bool kAdviceCommitmentsFlag = true;
  constexpr static bool kPermutationProductCommitmentsFlag = true;
  constexpr static bool kVanishingHPolyCommitmentsFlag = true;
  constexpr static bool kAdviceEvalsFlag = true;
  constexpr static bool kFixedEvalsFlag = true;
  constexpr static bool kCommonPermutationEvalsFlag = true;
  constexpr static bool kPermutationProductEvalsFlag = true;
  constexpr static bool kPermutationProductNextEvalsFlag = true;
  constexpr static bool kPermutationProductLastEvalsFlag = true;

  constexpr static size_t kN = 16;

  // clang-format off
  constexpr static std::string_view kPinnedConstraintSystem =
      "PinnedConstraintSystem { "
        "num_fixed_columns: 1, "
        "num_advice_columns: 2, "
        "num_instance_columns: 1, "
        "num_selectors: 1, "
        "gates: [Product("
          "Selector(Selector(0, true)), "
          "Sum("
            "Product("
              "Advice { "
                "query_index: 0, "
                "column_index: 0, "
                "rotation: Rotation(0) "
              "}, "
              "Advice { "
                "query_index: 1, "
                "column_index: 1, "
                "rotation: Rotation(0) "
              "}"
            "), "
            "Negated(Advice { "
              "query_index: 2, "
              "column_index: 0, "
              "rotation: Rotation(1) "
            "})"
          ")"
        ")], "
        "advice_queries: ["
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
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(1)"
          ")"
        "], "
        "instance_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Rotation(0)"
        ")], "
        "fixed_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Fixed "
          "}, "
          "Rotation(0)"
        ")], "
        "permutation: Argument { columns: ["
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Fixed "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 1, "
            "column_type: Advice "
          "}"
        "] }, "
        "lookups: [], "
        "constants: [Column { "
          "index: 0, "
          "column_type: Fixed "
        "}], "
        "minimum_degree: None "
      "}";
  // clang-format on

  constexpr static std::string_view kAssemblyFixedColumns[][kN] = {
      {
          "0x0000000000000000000000000000000000000000000000000000000000000007",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
  };

  constexpr static AnyColumnKey kAssemblyPermutationColumns[] = {
      InstanceColumnKey(0),
      FixedColumnKey(0),
      AdviceColumnKey(0),
      AdviceColumnKey(1),
  };

  // clang-format off
  constexpr static Label kCycleStoreMapping[][kN] = {
      {{2, 1}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 6}, {0, 0},  {2, 5},  {3, 0},  {2, 8},  {3, 2},  {1, 0},  {3, 4},
       {2, 4}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{2, 3}, {3, 1},  {2, 2},  {3, 3},  {2, 7},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };

  constexpr static Label kCycleStoreAux[][kN] = {
      {{2, 1}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 0}, {2, 1},  {2, 2},  {3, 0},  {2, 4},  {2, 2},  {2, 0},  {3, 4},
       {2, 4}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {2, 2},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };
  // clang-format on

  constexpr static size_t kCycleStoreSizes[][kN] = {
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  };

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true, false, true, false, true, false, false, false,
       false, false, false, false, false, false, false, false}
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
          "num_advice_columns: 2, "
          "num_instance_columns: 1, "
          "num_selectors: 1, "
          "gates: [Product("
            "Fixed { "
              "query_index: 1, "
              "column_index: 1, "
              "rotation: Rotation(0) "
            "}, "
            "Sum("
              "Product("
                "Advice { "
                  "query_index: 0, "
                  "column_index: 0, "
                  "rotation: Rotation(0) "
                "}, "
                "Advice { "
                  "query_index: 1, "
                  "column_index: 1, "
                  "rotation: Rotation(0) "
                "}"
              "), "
              "Negated(Advice { "
                "query_index: 2, "
                "column_index: 0, "
                "rotation: Rotation(1) "
              "})"
            ")"
          ")], "
          "advice_queries: ["
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
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(1)"
            ")"
          "], "
          "instance_queries: [("
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}, "
            "Rotation(0)"
          ")], "
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
          "permutation: Argument { columns: ["
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Fixed "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}"
          "] }, "
          "lookups: [], "
          "constants: ["
            "Column { "
              "index: 0, "
              "column_type: Fixed "
            "}"
          "], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x0f9cc629d0010671d7f755267acccfc8d5854f47d4e437ec84bddcc27b9f19d1, "
            "0x199b1dcc7aff518a4e49c6393bcf847cc3fde52ae69e326dea0d1d901424552a), "
          "(0x00253ad9f42498136d40e617f9d46d11bf33256914aab88e32fd413bed9baccc, "
            "0x154f7ddc2d7959e1abc7a0cada48f1b8f079c9fb45c6fdc2ee2121cc83a83b16)"
        "], "
        "permutation: VerifyingKey { commitments: ["
          "(0x2f8d8133413e0224e1cbb20aa8281458fe0d9cf4d723ff07ac0524acffc8be48, "
            "0x1f3e4dd4175e92cd38d3a8d3cd473e6daa6e9f28eb616222945904a6eadc52c6), "
          "(0x27f329c25618696f90d09ab41e80f4f2c1e927a6d0f86d8670e81882a449af36, "
            "0x1a69f0c1a2dee36915704c4fdee41ff8b7029c1894dc89356dd2cf4738e95df2), "
          "(0x2a7f86208a87462b85678f6aee3c9c3ee17371167b436ad58501819ae02cbffd, "
            "0x18b536eae002a69d57783fad5c63e35df0a2506dc43ee3d88053faff8f19cc40), "
          "(0x1bb17f6d2c12d9c8cd8860c00fcadebf63986bb330dfde5b9097b17047a4dfc1, "
            "0x1d6be3858a788f18e3c16ca8d3a88893f3678be9f827463af0f30d6b1f7df112)"
        "] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x012577899026da8b4a257e25b4edf52711038e19083900a458e1c1e18c29eb08";

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
          "0x0000000000000000000000000000000000000000000000000000000000000007",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
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
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
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
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
          "0x1b386c209eabea1777ad2736a8d8c1b4669d32a8c4784f51b62f1a2337000001",
      },
      {
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x1c691d91b8236e7d529d215f6002d508d558dfc8b053219170ed36d01d849247",
          "0x0f1f5883e65f820d14d56342dc92fd12a944d4cbbdce5377b7439bd07108fc9d",
          "0x1b8b7929b032ef27d92c2435436b46d14745adc1c4ce156fcf175f9593db2d7c",
          "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
          "0x2c2d581a99a701c15853e2260a74526e005e350e35cbc7dd18ffb7b2368d66f4",
          "0x2144f5eefad21e1ca37ae273a4ee5b4a7eef137cbbeb1d198c9e59c37ef70364",
          "0x20f1e8e5e94b190c36bf97fb364144b81191fc2ea3d0f6b121a9153eec12d94c",
          "0x27517fbd56f85221e5c138a4493917cbb0aa2cbae2e6ab760727978833000001",
          "0x1c691d91b8236e7d529d215f6002d508d558dfc8b053219170ed36d01d849247",
          "0x0f1f5883e65f820d14d56342dc92fd12a944d4cbbdce5377b7439bd07108fc9d",
          "0x1b8b7929b032ef27d92c2435436b46d14745adc1c4ce156fcf175f9593db2d7c",
          "0x2d5e098bb31e86271ccb415b196942d755b0a9c3f21dd9882fa3d63ab1000001",
          "0x2c2d581a99a701c15853e2260a74526e005e350e35cbc7dd18ffb7b2368d66f4",
          "0x2144f5eefad21e1ca37ae273a4ee5b4a7eef137cbbeb1d198c9e59c37ef70364",
          "0x20f1e8e5e94b190c36bf97fb364144b81191fc2ea3d0f6b121a9153eec12d94c",
      },
  };

  constexpr static std::string_view kPermutationsColumns[][kN] = {
      {
          "0x08e7cbfea108224b0777f0558503af41585b75ab8d4d807505158f4bc8c771de",
          "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
          "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
          "0x107aab49e65a67f9da9cd2abf78be38bd9dc1d5db39f81de36bcfa5b4b039043",
          "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
          "0x2290ee31c482cf92b79b1944db1c0147635e9004db8c3b9d13644bef31ec3bd3",
          "0x1d59376149b959ccbd157ac850893a6f07c2d99b3852513ab8d01be8e846a566",
          "0x2d8040c3a09c49698c53bfcb514d55a5b39e9b17cb093d128b8783adb8cbd723",
          "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
          "0x0f5c21d0ca65e0db9be1f670eca407d08ec5ec67626a74f892ccebcd0cf9b9f6",
          "0x0530d09118705106cbb4a786ead16926d5d174e181a26686af5448492e42a181",
          "0x1fe9a328fad7382fddb3730a89f574d14e57caeac619eeb30d24fb38a4fc6fbe",
          "0x0000000000000000b3c4d79d41a91758cb49c3517c4604a520cff123608fc9cb",
          "0x0dd360411caed09700b52c71a6655715c4d558439e2d34f4307da9a4be13c42e",
          "0x130b17119778465cfb3acaee30f81dee20710ead41671f568b11d9ab07b95a9b",
          "0x02e40daf409556c02bfc85eb303402b774954d30aeb0337eb85a71e6373428de",
      },
      {
          "0x13b360d4e82fe915fed16081038f98c211427b87a281bd733c277dbadf10372b",
          "0x133f51f46c2a51201fbb89dc1d6868acfef0f7ce0e572ccb98e04e99eac3ea36",
          "0x1240a374ee6f71e12df10a129946ca9ba824f58a50e3a58b4c68dd5590b74ad8",
          "0x009553089042fe83ab570b54e4e64d307d8e8b20d568e782ece3926981d0a96c",
          "0x14a6c152ace4b16a42e1377e400798e15ac60320c21d90277890dbdd551d1912",
          "0x035992598be4d2ae5334f24f30355a1fca7f0e28762a178d79e993dca70eaabf",
          "0x03b645319eb70d7a692ea8c87fbcab1f292cd502071d9ffb7291ebe28356aa07",
          "0x231e38741f5c3f0ce2509bd3289b09a893b14b136eea6ed00cec9ac9271c9db5",
          "0x2741e304be6aaf5f53641f0bacb8e9ebccd45eba1b23316bbcd39ed80acc165f",
          "0x1d24fc7e75074f099894bbda6418efb02942f07a6b6243c5ab01a6fa053c15cb",
          "0x1e23aafdf2c22e488a5f3ba3e83a8dc1800ef2be28d5cb05f779183e5f48b529",
          "0x2fcefb6a50eea1a60cf93a619c9b0b2caaa55d27a450890e56fe632a6e2f5695",
          "0x1bbd8d20344ceebf756f0e384179bf7bcd6de527b79be069cb5119b69ae2e6ef",
          "0x2d0abc19554ccd7b651b5367514bfe3d5db4da20038f5903c9f861b748f15542",
          "0x2cae0941427a92af4f219cee01c4ad3dff071346729bd095d15009b16ca955fa",
          "0x0d4615fec1d5611cd5ffa9e358e64eb494829d350acf01c136f55acac8e3624c",
      },
      {
          "0x2e39c17d3e15071a9341cfcea233ad5f25a14dea28716de88ae369ac2c9944ac",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x04765b5102d6627cfcc389436e96bbc9aa478dcb720b546c77063077a40910ce",
          "0x18afdf23e9bd9302673fc1e076a492d4d65bd18ebc4d854ed189139bab313e52",
          "0x1cb0ed9df901b713b97ee5357df1bf9b16f16cc0d737b31e07ba77d910efc8d6",
          "0x034183d253b6b250dae0a457797029523434d2b6bc41c09b6ef409bd970e4208",
          "0x09226b6e22c6f0ca64ec26aad4c86e715b5f898e5e963f25870e56bbe533e9a2",
          "0x2f549305063b1803a77ba92486c5383e00468fc857ec66d944868c4a309e596a",
          "0x086398ace043cf0db4e99f2712f392ddbd9c7b9202b2fcadf26e9b145316df9f",
          "0x277c827440297ddeb0d85560fc7da91bcfd8729cec6bf01c3ecc664827388e23",
          "0x24f4c8596963484c0569feb1f2a79f19eb87843cd95fd26af5bdb12c8a26ea2e",
          "0x096b10d95e053da3dea44cf0c8ea6de7e962b0f71046aab3779baa3d2b772a01",
          "0x2800b5c600edd11c0366a68f6e8dc57f6a976cb6770673e351735a7f9ce92062",
          "0x2bedf321de5b3dacbb8cbc7312ea9c937dec5a7d07ae1c24ccdbc51c4bf6ef33",
          "0x022a8cf5a31c990f250e75e7df4daafe02929a5e514802a8b8fe8be7c366bb55",
          "0x06272e83847dd219527e22d89b87a6efdf7fc2b0f32d564762853d937378875d",
      },
      {
          "0x26f93d99832c6285d9abf8c5b896ea753ed137516972c5ddcc464b56c488d600",
          "0x2f0e061e83e8b12c1bdf8df7cca02483295e89f78a462ce38859661e06501815",
          "0x0b6f861977ce57ddb2e647048ed9b9433cac640ba0599e264e24446765d915d3",
          "0x0f34fda7cd268bf095354ea6c2631826c349d7518bf094361bc91f61c519c7fb",
          "0x2a3d1fef5cb3ce1065d222dde5f9b16d48b42597868c1a49e15cb8007c8778a4",
          "0x1e626bf9ef3c8920522383ff9be21287c3af8d47fe61ff9af6c6d8d0118154bc",
          "0x02898db49f7022cac0aeabf721a3c7ab96e14ad35e7d6b0410b868ff504b62fc",
          "0x2a150ee7450ad90cace368ea424c8fdf36b3821e79f5f1c617a5f8e74c19b4cf",
          "0x17b46f4ef7740d27511083d60adcc58851d816b9bd6beb427258e1f844cec1af",
          "0x015648545d48eefd9c70b7beb4e133d9fed55e50ef7343adbb888f75e9afe7ec",
          "0x2d22caa08d7aedd8dd6fa15f08112f0af3ff1591bd77aff5d4edebd658f1bdf9",
          "0x212f50cb140b1439231af70fbf1e403664ea10f6edc8dc5b2818d6322ae63806",
          "0x010fbb6ddaf6882610d49c91fabc201f27ed588021cd09b7ff5b6949bf61a697",
          "0x1201e278f1f51709662cc1b6e59f45d564845b007b5770f64d1b1cc3de7eab45",
          "0x2ddac0be41c17d5ef7a199bf5fdd90b191529d751b3c058d33298c949fb49d05",
          "0x064f3f8b9c26c71d0b6cdccc3f34c87df1806629ffc37ecb2c3bfcaca3e64b32",
      },
  };

  constexpr static std::string_view kPermutationsPolys[][kN] = {
      {
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771f",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
          "0x09a14b757449d02c83068c1790987b858d0f72e84fa79d228d0bb700798c771e",
      },
      {
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x1ef75d16d1a336615f2d98c8105d77bf28546e4da8161849f012c49c8dd1ae7b",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
          "0x15d4f1a8aedc4596fa41721d3b95094dccf4e4bf497fd92469046de0a89dc4d9",
      },
      {
          "0x18eae12fd704e5bc3d0acf5f32330e0f3a23a3600b66f24ce31ff4dcd7950b5f",
          "0x1668c537404f9338905d9b885be857f7927a91f29c6b623e5d744fb232c8c7cb",
          "0x0d1bfa0c9869dfa02919a0d7834f3f2a4d64218aa8582ca4b43c029a435d42f1",
          "0x2ceb4c0d1b5b06357be3af62312632f070a6eac793edb5e815c743e973aaab7d",
          "0x2c814b07e5b0482b27173e2370c3a08662204993e2d8fdb097c17c0a93fb2ebd",
          "0x11656a996c4ae9762c3899876dbeeef8c9638869fa4a66dfe73e4b59685040a3",
          "0x1753fa2f3bf232558b65d8726f2ce1d9b092400c892a93d51017051b72c4aa7e",
          "0x231490648c345450dd40f890419598f601d543b3e319fcacf23a9d9ecb819762",
          "0x092f15a06a9434ad833065f6c50caf3d5afa51cf219ce91f1e429bf444f56a50",
          "0x0b7e67f7e58f3f373c2e9a9cae2e7c1b3a9b8571964ae0cab3e7568106eaa6e0",
          "0x2053b23f55828893e49b77ac55a9d0587fa75004919673d04e07bf0940e88120",
          "0x2f56ca1e5a468ea9c03068b5f43d501484a981ddfde12ad4f4baa97b2144f9f8",
          "0x0cb5df9af846bd3540562259624801f754344c5466a397fef1565be386dc9ef6",
          "0x19ac12db90e5885a5224538b937cbcf1298034e8dde3eda38e033b460ce6afa9",
          "0x2498f7b8d9eb6152686b24b08d67043db5445508860199e20f6aae67d96b6038",
          "0x205f2438ff62bf51c8571e289c1a7eec31cc7961b6d43e347a59808f956596bd",
      },
      {
          "0x28677e97c83844d3b64dbdd904a21dd7bc3fd5894ee4788eae93d169e640b8ac",
          "0x0b09ea89c6731047056d4cde1f39228a9a88b04ffa4b7db592540476c69a20d5",
          "0x22c644b2532a4b71b8545dde2743e1d1b73b6b0426942d10912538e47312096d",
          "0x118e682968bef69c68c23668dfb585f320bad9da0b5d89d01a581bd198d8779d",
          "0x09230d4716f64c08081addf15104149f62107231664d96828840909836675e32",
          "0x296a7370fa3f06b11a95a596e8a8b260ea0a6ed3aa2810059fd557dda2ed0369",
          "0x283c4b2b66fee7fbb2a6c2591755de8829136b66ffd2ad6d206a98fa0e9bc5a3",
          "0x1ed5e6497872a98d4f8e0f4da1cbd26a07790e6e6e5be6c12989d9c257278864",
          "0x28677e97c83844d3b64dbdd904a21dd7bc3fd5894ee4788eae93d169e640b8ac",
          "0x22be59d8bde71d6e567dd0b42a15e812ec60c709b7b768f804ace66f0b68e284",
          "0x22c644b2532a4b71b8545dde2743e1d1b73b6b0426942d10912538e47312096d",
          "0x118e682968bef69c68c23668dfb585f320bad9da0b5d89d01a581bd198d8779d",
          "0x09230d4716f64c08081addf15104149f62107231664d96828840909836675e32",
          "0x296a7370fa3f06b11a95a596e8a8b260ea0a6ed3aa2810059fd557dda2ed0369",
          "0x283c4b2b66fee7fbb2a6c2591755de8829136b66ffd2ad6d206a98fa0e9bc5a3",
          "0x1ed5e6497872a98d4f8e0f4da1cbd26a07790e6e6e5be6c12989d9c257278864",
      },
  };

  constexpr static uint8_t kProof[] = {
      19,  29,  49,  75,  8,   191, 254, 126, 107, 125, 58,  193, 235, 208, 41,
      102, 196, 144, 108, 7,   165, 223, 43,  109, 53,  216, 237, 43,  184, 227,
      83,  9,   184, 235, 153, 217, 206, 69,  132, 165, 210, 205, 116, 237, 238,
      162, 49,  49,  204, 46,  223, 58,  73,  229, 128, 143, 213, 188, 0,   212,
      5,   85,  163, 5,   19,  29,  49,  75,  8,   191, 254, 126, 107, 125, 58,
      193, 235, 208, 41,  102, 196, 144, 108, 7,   165, 223, 43,  109, 53,  216,
      237, 43,  184, 227, 83,  9,   184, 235, 153, 217, 206, 69,  132, 165, 210,
      205, 116, 237, 238, 162, 49,  49,  204, 46,  223, 58,  73,  229, 128, 143,
      213, 188, 0,   212, 5,   85,  163, 5,   127, 225, 164, 3,   197, 222, 99,
      117, 232, 31,  74,  110, 227, 32,  48,  192, 153, 241, 171, 52,  191, 26,
      68,  45,  197, 13,  113, 23,  246, 13,  231, 166, 64,  66,  93,  243, 151,
      203, 112, 88,  120, 207, 230, 170, 173, 254, 252, 207, 63,  239, 95,  136,
      10,  226, 56,  39,  154, 39,  76,  240, 18,  10,  142, 28,  151, 164, 103,
      32,  254, 100, 230, 86,  82,  1,   70,  1,   180, 136, 218, 38,  91,  237,
      148, 213, 248, 198, 195, 125, 183, 139, 13,  25,  173, 78,  126, 142, 94,
      35,  73,  89,  164, 177, 165, 40,  196, 91,  142, 100, 101, 174, 88,  101,
      62,  118, 77,  113, 64,  35,  18,  251, 201, 60,  109, 41,  87,  12,  233,
      144, 194, 190, 107, 221, 112, 231, 78,  27,  96,  65,  41,  108, 62,  155,
      56,  146, 132, 59,  154, 117, 155, 164, 168, 171, 246, 216, 93,  239, 223,
      204, 133, 2,   39,  160, 123, 232, 7,   238, 13,  107, 196, 241, 124, 154,
      169, 67,  116, 166, 56,  151, 214, 194, 186, 125, 118, 163, 199, 247, 59,
      24,  208, 119, 122, 163, 116, 241, 113, 160, 218, 185, 1,   57,  204, 40,
      197, 69,  109, 184, 245, 200, 155, 210, 27,  184, 180, 198, 202, 51,  4,
      84,  171, 139, 243, 38,  69,  151, 122, 54,  14,  56,  136, 105, 156, 65,
      234, 163, 12,  136, 101, 96,  225, 135, 193, 103, 253, 5,   183, 161, 240,
      107, 76,  93,  63,  255, 193, 89,  223, 140, 1,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   103, 21,  209, 146,
      1,   82,  113, 58,  64,  67,  95,  172, 141, 84,  72,  42,  229, 27,  151,
      165, 6,   201, 90,  138, 39,  146, 238, 171, 253, 188, 2,   142, 65,  69,
      245, 48,  124, 53,  52,  41,  169, 118, 173, 194, 123, 190, 197, 89,  126,
      206, 19,  99,  193, 187, 33,  64,  235, 33,  161, 242, 64,  224, 182, 28,
      232, 144, 93,  25,  237, 225, 238, 156, 100, 254, 231, 202, 148, 15,  107,
      121, 62,  162, 160, 226, 233, 110, 146, 4,   107, 126, 136, 195, 38,  162,
      61,  11,  204, 20,  246, 68,  156, 199, 230, 119, 41,  240, 125, 4,   175,
      162, 29,  6,   233, 171, 69,  177, 59,  119, 202, 32,  119, 74,  195, 20,
      71,  46,  58,  15,  43,  106, 225, 52,  175, 228, 9,   113, 36,  132, 251,
      50,  157, 212, 147, 80,  135, 1,   153, 128, 227, 208, 221, 153, 4,   6,
      147, 247, 174, 180, 129, 14,  232, 144, 93,  25,  237, 225, 238, 156, 100,
      254, 231, 202, 148, 15,  107, 121, 62,  162, 160, 226, 233, 110, 146, 4,
      107, 126, 136, 195, 38,  162, 61,  11,  204, 20,  246, 68,  156, 199, 230,
      119, 41,  240, 125, 4,   175, 162, 29,  6,   233, 171, 69,  177, 59,  119,
      202, 32,  119, 74,  195, 20,  71,  46,  58,  15,  43,  106, 225, 52,  175,
      228, 9,   113, 36,  132, 251, 50,  157, 212, 147, 80,  135, 1,   153, 128,
      227, 208, 221, 153, 4,   6,   147, 247, 174, 180, 129, 14,  142, 196, 63,
      34,  8,   2,   236, 221, 118, 2,   32,  13,  252, 166, 9,   145, 155, 143,
      102, 87,  176, 241, 131, 217, 131, 153, 128, 123, 100, 68,  113, 39,  56,
      59,  88,  212, 132, 43,  80,  80,  37,  252, 240, 215, 66,  33,  244, 204,
      123, 180, 101, 62,  147, 229, 119, 148, 44,  47,  148, 95,  174, 78,  83,
      38,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   44,  186, 67,  191, 236, 7,   211, 217, 169, 158, 174, 166,
      95,  237, 145, 220, 157, 191, 24,  173, 91,  163, 2,   200, 171, 0,   1,
      53,  138, 1,   245, 11,  182, 224, 189, 181, 200, 187, 146, 227, 182, 188,
      15,  122, 71,  167, 113, 161, 114, 112, 233, 133, 169, 77,  167, 32,  163,
      114, 130, 13,  56,  30,  155, 8,   11,  254, 248, 201, 168, 128, 166, 180,
      238, 139, 27,  248, 22,  63,  236, 216, 177, 33,  29,  4,   140, 225, 157,
      127, 31,  92,  70,  103, 251, 233, 142, 32,  152, 76,  190, 44,  81,  109,
      249, 211, 91,  97,  187, 223, 199, 161, 96,  71,  36,  224, 67,  110, 244,
      243, 210, 104, 60,  247, 95,  236, 28,  254, 54,  1,   213, 221, 250, 122,
      212, 75,  9,   33,  153, 25,  10,  37,  210, 5,   8,   142, 138, 210, 147,
      191, 223, 194, 177, 166, 122, 215, 220, 20,  242, 45,  129, 4,   161, 99,
      235, 177, 225, 132, 9,   80,  54,  196, 18,  88,  39,  126, 31,  240, 223,
      255, 126, 92,  55,  141, 8,   220, 239, 201, 149, 213, 115, 140, 232, 7,
      120, 84,  236, 42,  248, 14,  141, 0,   5,   50,  135, 194, 233, 85,  16,
      2,   232, 91,  220, 83,  33,  22,  176, 113, 174, 193, 230, 98,  176, 183,
      227, 42,  40,  74,  92,  28,  234, 158, 50,  106, 46,  37,  185, 160, 129,
      227, 84,  45,  124, 145, 236, 111, 45,  101, 176, 249, 147, 58,  137, 55,
      251, 58,  172, 8,   53,  217, 33,  179, 247, 171, 33,  154, 230, 219, 102,
      98,  51,  94,  42,  222, 177, 197, 235, 8,   128, 33,  106, 181, 141, 53,
      45,  62,  223, 217, 193, 13,  97,  58,  116, 65,  181, 178, 45,  136, 124,
      144, 79,  119, 186, 107, 198, 163, 108, 15,  22,  134, 226, 153, 88,  102,
      52,  121, 185, 189, 205, 123, 75,  19,  192, 203, 139, 211, 156, 110, 67,
      191, 174, 243, 221, 136, 44,  20,  126, 162, 18,  227, 98,  160, 188, 217,
      234, 18,  20,  104, 214, 254, 40,  112, 211, 18,  122, 245, 183, 30,  198,
      129, 160, 62,  219, 111, 140, 121, 38,  57,  214, 37,  223, 154, 190, 249,
      124, 195, 141, 142, 236, 188, 199, 101, 162, 65,  234, 24,  96,  181, 126,
      246, 252, 107, 26,  142, 140, 101, 123, 104, 216, 84,  11,  113, 221, 144,
      66,  96,  50,  225, 185, 255, 201, 18,  223, 192, 222, 128, 79,  16,  125,
      89,  221, 50,  95,  246, 91,  196, 74,  6,   119, 180, 172, 85,  157, 210,
      196, 227, 140, 43,  57,  115, 40,  252, 155, 118, 40,  59,  164, 243, 234,
      39,  10,  236, 201, 162, 65,  195, 134, 15,  147, 117, 241, 184, 28,  8,
      189, 70,  207, 161, 188, 206, 167, 11,  33,  188, 238, 37,  121, 146, 32,
      153, 13,  45,  67,  30,  159, 171, 62,  53,  42,  44,  135, 15,  34,  48,
      186, 181, 40,  54,  17,  2,   11,  26,  49,  48,  126, 76,  104, 57,  151,
      38,  59,  87,  46,  19,  191, 84,  121, 223, 179, 241, 158, 143, 0,   183,
      159, 67,  65,  137, 109, 224, 180, 218, 133, 51,  200, 152, 77,  76,  8,
      134, 207, 91,  31,  102, 230, 21,  124, 131, 21,  228, 218, 145, 179, 121,
      144, 161, 120, 160, 219, 220, 197, 177, 46,  16,  184, 39,  160, 31,  150,
      232, 255, 1,   172, 7,   192, 47,  147, 13,  112, 126, 39,  227, 226, 77,
      246, 2,   127, 76,  4,   65,  20,  72,  174, 226, 9,   85,  222, 18,  119,
      57,  61,  127, 160, 120, 159, 64,  95,  142, 206, 27,  177, 71,  186, 98,
      62,  77,  221, 242, 183, 216, 211, 123, 14,  46,  235, 88,  55,  121, 13,
      192, 250, 25,  4,   229, 43,  45,  40,  141, 98,  1,   118, 25,  20,  183,
      51,  17,  131, 13,  80,  92,  203, 71,  35,  30,  242, 124, 253, 1,   145,
      242, 0,   72,  5,   232, 37,  235, 253, 114, 217, 100, 152, 94,  103, 40,
      4,   85,  204, 35,  109, 114, 118, 5,   34,  214, 2,   182, 236, 47,  124,
      96,  230, 40,  13,  171, 214, 160, 183, 205, 171, 53,  2,   200, 225, 0,
      12,  8,   139, 247, 90,  9,   250, 149, 235, 35,  101, 92,  2,   48,  135,
      6,   210, 146, 7,   143, 136, 217, 119, 66,  116, 22,  66,  151, 77,  102,
      125, 36,  62,  24,  159, 100, 74,  215, 27,  51,  90,  186, 10,  219, 218,
      21,  218, 206, 157, 27,  62,  68,  250, 70,  205, 164, 34,  215, 124, 203,
      45,  187, 171, 131, 36,  1,   106, 159, 99,  124, 92,  119, 72,  9,   156,
      231, 150, 98,  128, 74,  20,  148, 152, 87,  89,  197, 87,  249, 99,  188,
      30,  171, 156, 61,  250, 148, 254, 37,  239, 44,  64,  123, 239, 75,  100,
      82,  196, 227, 147, 7,   251, 34,  5,   66,  222, 147, 110, 235, 124, 40,
      45,  83,  188, 74,  140, 109, 254, 39,  178, 29,  2,   74,  21,  36,  248,
      8,   246, 86,  22,  104, 204, 126, 222, 205, 217, 61,  34,  27,  76,  75,
      209, 54,  81,  170, 235, 3,   40,  82,  255, 71,  214, 15,  195, 232, 94,
      47,  170, 193, 202, 105, 194, 8,   140, 170, 93,  209, 106, 51,  83,  97,
      19,  62,  235, 15,  93,  126, 44,  134, 222, 205, 194, 84,  181, 33};

  // clang-format off
  constexpr static Point kAdviceCommitments[][2] = {
      {
          {"0x0953e3b82bedd8356d2bdfa5076c90c46629d0ebc13a7d6b7efebf084b311d13",
            "0x0e1846fc46b7f84859cf41eabe2cfadf1c08f6be2df3fd75f47dd9945ecdea66"},
          {"0x05a35505d400bcd58f80e5493adf2ecc3131a2eeed74cdd2a58445ced999ebb8",
            "0x167f11ce9f3d3ebc24fe1a9d722ffdc7ee94a1734d544837868c7ccd7de960f0"},
      },
      {
          {"0x0953e3b82bedd8356d2bdfa5076c90c46629d0ebc13a7d6b7efebf084b311d13",
            "0x0e1846fc46b7f84859cf41eabe2cfadf1c08f6be2df3fd75f47dd9945ecdea66"},
          {"0x05a35505d400bcd58f80e5493adf2ecc3131a2eeed74cdd2a58445ced999ebb8",
            "0x167f11ce9f3d3ebc24fe1a9d722ffdc7ee94a1734d544837868c7ccd7de960f0"},
      },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x1bbbaa33ab11f8073b8fd7cd635376bf76535666874fed4eda0aa27081c7224d";

  constexpr static std::string_view kBeta =
      "0x2ea3ec52fc8914364d92275b8e2cc142ff57c1833be6ff1010c0afcb49e418f5";

  constexpr static std::string_view kGamma =
      "0x1b9c1308750907b62d2039a8a1ab39aef60f44d5c0f983c8996a02f282eb6a67";

  // clang-format off
  constexpr static Point kPermutationProductCommitments[][4] = {
      {
          {"0x26e70df617710dc52d441abf34abf199c03020e36e4a1fe87563dec503a4e17f",
           "0x12b60345e427c9b7bedabe0582f8ea1757afb34a2aeb7221c111cb809a58b645"},
          {"0x1c8e0a12f04c279a2738e20a885fef3fcffcfeadaae6cf785870cb97f35d4240",
           "0x00cbb8c3b8d1c13de7fbb52cce108e8eac9d25bfdfbe1d34e7f9bda4b5ba493a"},
          {"0x0e7e4ead190d8bb77dc3c6f8d594ed5b26da88b40146015256e664fe2067a497",
           "0x2bec88861fe25d0e544588e1ce1a37dbce9bcbca2d36a6d207806c6feda7ebcb"},
          {"0x10e90c57296d3cc9fb122340714d763e6558ae65648e5bc428a5b1a45949235e",
           "0x10b67dc003e08b6cc40bae249415357be1bf15c198f4e717f6d2b709a4b5ca35"},
      },
      {
          {"0x0285ccdfef5dd8f6aba8a49b759a3b8492389b3e6c2941601b4ee770dd6bbec2",
           "0x18c3f0b71002cdda9e5896851e5bd3ab3d4739abf891dccb8680e5c9115ee004"},
          {"0x237a77d0183bf7c7a3767dbac2d69738a67443a99a7cf1c46b0dee07e87ba027",
           "0x1d66dfa81519f0b2e109ccda3604ac8ad0bcfc22bc5b4817da3b54dde01e9545"},
          {"0x174526f38bab540433cac6b4b81bd29bc8f5b86d45c528cc3901b9daa071f174",
           "0x2097a71126e3deec5aa3f47ed4c382f86589bd40eb10a2fa9451b08be5455d79"},
          {"0x0cdf59c1ff3f5d4c6bf0a1b705fd67c187e16065880ca3ea419c6988380e367a",
           "0x2310ba4c689be5bf70c94a52c681713c0e225c615e9dcf0e3d61827921244023"},
      },
  };
  // clang-format on

  constexpr static std::string_view kY =
      "0x144fee08a5557e8bbd9a9ec6e9f5e9ce0b65701a0549dc6b6edacf18414667c3";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x0e02bcfdabee92278a5ac906a5971be52a48548dac5f43403a71520192d11567",
       "0x253356792d038018c8b5db38d611b214d18ad138a02091569c8e38460f5f094f"},
      {"0x1cb6e040f2a121eb4021bbc16313ce7e59c5be7bc2ad76a92934357c30f54541",
       "0x068c8fbf67dc71a3a2ee286150c72c67cc26d5b9286757552f8832fe751d8b26"},
  };

  constexpr static std::string_view kX =
      "0x2e04b4bdc49c7acd56eabd66f8d404477dbf0e8f0f928b7aea12d312c3bc9421";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x0b3da226c3887e6b04926ee9e2a0a23e796b0f94cae7fe649ceee1ed195d90e8",
          "0x0f3a2e4714c34a7720ca773bb145abe9061da2af047df02977e6c79c44f614cc",
          "0x0e81b4aef793060499ddd0e3809901875093d49d32fb84247109e4af34e16a2b",
      },
      {
          "0x0b3da226c3887e6b04926ee9e2a0a23e796b0f94cae7fe649ceee1ed195d90e8",
          "0x0f3a2e4714c34a7720ca773bb145abe9061da2af047df02977e6c79c44f614cc",
          "0x0e81b4aef793060499ddd0e3809901875093d49d32fb84247109e4af34e16a2b",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x277144647b809983d983f1b057668f9b9109a6fc0d200276ddec0208223fc48e",
      "0x26534eae5f942f2c9477e5933e65b47bccf42142d7f0fc2550502b84d4583b38",
  };

  constexpr static std::string_view kCommonPermutationEvals[] = {
      "0x0bf5018a350100abc802a35bad18bf9ddc91ed5fa6ae9ea9d9d307ecbf43ba2c",
      "0x089b1e380d8272a320a74da985e97072a171a7477a0fbcb6e392bbc8b5bde0b6",
      "0x208ee9fb67465c1f7f9de18c041d21b1d8ec3f16f81b8beeb4a680a8c9f8fe0b",
      "0x0136fe1cec5ff73c68d2f3f46e43e0244760a1c7dfbb615bd3f96d512cbe4c98",
  };

  constexpr static std::string_view kPermutationProductEvals[][4] = {
      {
          "0x04812df214dcd77aa6b1c2dfbf93d28a8e0805d2250a199921094bd47afaddd5",
          "0x08ac3afb37893a93f9b0652d6fec917c2d54e381a0b9252e6a329eea1c5c4a28",
          "0x12d37028fed6681412ead9bca062e312a27e142c88ddf3aebf436e9cd38bcbc0",
          "0x27eaf3a43b28769bfc2873392b8ce3c4d29d55acb477064ac45bf65f32dd597d",
      },
      {
          "0x132e573b269739684c7e30311a0b02113628b5ba30220f872c2a353eab9f1e43",
          "0x1bce8e5f409f78a07f3d397712de5509e2ae481441044c7f02f64de2e3277e70",
          "0x080c00e1c80235abcdb7a0d6ab0d28e6607c2fecb602d6220576726d23cc5504",
          "0x25fe94fa3d9cab1ebc63f957c559579894144a806296e79c0948775c7c639f6a",
      },
  };
  constexpr static std::string_view kPermutationProductNextEvals[][4] = {
      {
          "0x07e88c73d595c9efdc088d375c7effdff01f7e275812c436500984e1b1eb63a1",
          "0x0dc1d9df3e2d358db56a218008ebc5b1de2a5e336266dbe69a21abf7b321d935",
          "0x18ea41a265c7bcec8e8dc37cf9be9adf25d63926798c6fdb3ea081c61eb7f57a",
          "0x2d0d9920927925eebc210ba7cebca1cf46bd081cb8f175930f86c341a2c9ec0a",
      },
      {
          "0x15e6661f5bcf86084c4d98c83385dab4e06d8941439fb7008f9ef1b3df7954bf",
          "0x197601628d282d2be50419fac00d793758eb2e0e7bd3d8b7f2dd4d3e62ba47b1",
          "0x183e247d664d974216744277d9888f0792d2068730025c6523eb95fa095af78b",
          "0x1db227fe6d8c4abc532d287ceb6e93de420522fb0793e3c452644bef7b402cef",
      },
  };

  constexpr static std::string_view kPermutationProductLastEvals[][4] = {
      {
          "0x2ae3b7b062e6c1ae71b0162153dc5be8021055e9c2873205008d0ef82aec5478",
          "0x134b7bcdbdb97934665899e286160f6ca3c66bba774f907c882db2b541743a61",
          "0x104f80dec0df12c9ffb9e132604290dd710b54d8687b658c8e1a6bfcf67eb560",
          "",
      },
      {
          "0x0d932fc007ac01ffe8961fa027b8102eb1c5dcdba078a19079b391dae415837c",
          "0x28675e9864d972fdeb25e8054800f29101fd7cf21e2347cb5c500d831133b714",
          "0x012483abbb2dcb7cd722a4cd46fa443e1b9dceda15dadb0aba5a331bd74a649f",
          "",
      },
  };

  constexpr static std::string_view kHEval =
      "0x2af439530de48c1318f0ecc2905916d3d1b0dfcccdf3dd2a3eeefd6cffae9f62";

  static void TestConfig(FieldConfig<F>& config) {
    std::array<AdviceColumnKey, 2> expected_advice = {
        AdviceColumnKey(0),
        AdviceColumnKey(1),
    };
    EXPECT_EQ(config.advice(), expected_advice);
    EXPECT_EQ(config.instance(), InstanceColumnKey(0));
    EXPECT_EQ(config.s_mul(), Selector::Simple(0));
  }

  static Circuit GetCircuit() {
    F constant(7);
    F a(2);
    F b(3);
    return Circuit(std::move(constant), std::move(a), std::move(b));
  }

  static std::vector<Circuit> Get2Circuits() {
    Circuit circuit = GetCircuit();
    return {circuit, std::move(circuit)};
  }

  static std::vector<Evals> GetInstanceColumns() {
    F constant(7);
    F a(2);
    F b(3);
    F c = std::move(constant) * std::move(a).Square() * std::move(b).Square();
    std::vector<F> instance_column = {std::move(c)};
    return {Evals(std::move(instance_column))};
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_SIMPLE_CIRCUIT_TEST_DATA_H_

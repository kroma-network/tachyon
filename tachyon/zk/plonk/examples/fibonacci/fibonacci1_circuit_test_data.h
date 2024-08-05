#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/examples/circuit_test_data.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PCS, typename LS, typename SFINAE = void>
class Fibonacci1TestData : public CircuitTestData<Circuit, PCS, LS> {};

// FloorPlanner = SimpleFloorPlanner
template <typename Circuit, typename PCS, typename LS>
class Fibonacci1TestData<Circuit, PCS, LS,
                         std::enable_if_t<IsSimpleFloorPlanner<Circuit>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;

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
        "num_fixed_columns: 0, "
        "num_advice_columns: 3, "
        "num_instance_columns: 1, "
        "num_selectors: 1, "
        "gates: [Product("
          "Selector(Selector(0, true)), "
          "Sum("
            "Sum("
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
              "column_index: 2, "
              "rotation: Rotation(0) "
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
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          ")"
        "], "
        "instance_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Rotation(0)"
        ")], "
        "fixed_queries: [], "
        "permutation: Argument { columns: ["
          "Column { "
            "index: 0, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 1, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 2, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}"
        "] }, "
        "lookups: [], "
        "constants: [], "
        "minimum_degree: None "
      "}";
  // clang-format on

  constexpr static AnyColumnKey kAssemblyPermutationColumns[] = {
      AdviceColumnKey(0),
      AdviceColumnKey(1),
      AdviceColumnKey(2),
      InstanceColumnKey(0),
  };

  // clang-format off
  constexpr static Label kCycleStoreMapping[][kN] = {
      {{3, 0}, {3, 1}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},
      {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{0, 1}, {2, 0}, {2, 1},  {2, 2},  {2, 3},  {2, 4},  {2, 5},  {2, 6},
      {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{0, 2}, {0, 3}, {0, 4},  {0, 5},  {0, 6},  {0, 7},  {1, 7},  {3, 2},
      {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 0}, {1, 0}, {2, 7},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
      {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };

  constexpr static Label kCycleStoreAux[][kN] = {
      {{0, 0}, {1, 0},  {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},
        {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{1, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
        {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{1, 1}, {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},  {2, 7},
        {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 0}, {1, 0},  {2, 7},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
        {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };
  // clang-format on

  constexpr static size_t kCycleStoreSizes[][kN] = {
      {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  };

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true,  true,  true,  true,  true,  true,  true,  true,
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
          "num_fixed_columns: 1, "
          "num_advice_columns: 3, "
          "num_instance_columns: 1, "
          "num_selectors: 1, "
          "gates: [Product("
            "Fixed { "
              "query_index: 0, "
              "column_index: 0, "
              "rotation: Rotation(0) "
            "}, "
            "Sum("
              "Sum("
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
                "column_index: 2, "
                "rotation: Rotation(0) "
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
                "index: 2, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
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
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}"
          "] }, "
          "lookups: [], "
          "constants: [], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x297ff8a661d1fa1196c065b6fb7df901fb16b5b83168ddca6c7749d963cc967a, "
            "0x1a7b1f2b5f3e35fc4c706ece6b8f646e95904f9aa417f8a31fd37a41da167ec1)"
        "], "
        "permutation: VerifyingKey { commitments: ["
          "(0x28472c5c287afbed2c1cbad418f7736db3be191e735e933e7531f790cc0b454b, "
            "0x2553db7d81ee798baa1b89b83e3cfd5516872410ac1ef7ff8e97801a32f54b22), "
          "(0x1881f5e53bde69bb20eec1af0e8b17f23f3e49cf6929f13417a5dc389f70acec, "
            "0x19e1cd19519ba4353b95ba5a16a10479eec522952c97acd31199eb96b558b158), "
          "(0x1a899815b7cb019ce89ac7da39c4c0e071bd042ebffe13965de4e9e1cb968e17, "
            "0x1c2bc1e987ea49a29fa8a6e2168541a29fc27b0150b2b85f8bb2d76e1abd029b), "
          "(0x02f5f4023c8be80b0fe6ff231b1c292527881597645688da1cbfe83be61bfdc9, "
            "0x0eb912c0ac4f39a2e76c0fb7ac2d3977a4cc909db67680eb9e3c2436f4dc85ae)"
        "] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x0e149c09b16d13bdc8a09508e1dab4af7399ebe708e0fc37a7fd59d43974596f";

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
  };

  constexpr static std::string_view kPermutationsColumns[][kN] = {
      {
          "0x18afdf23e9bd9302673fc1e076a492d4d65bd18ebc4d854ed189139bab313e52",
          "0x2f0e061e83e8b12c1bdf8df7cca02483295e89f78a462ce38859661e06501815",
          "0x133f51f46c2a51201fbb89dc1d6868acfef0f7ce0e572ccb98e04e99eac3ea36",
          "0x1240a374ee6f71e12df10a129946ca9ba824f58a50e3a58b4c68dd5590b74ad8",
          "0x009553089042fe83ab570b54e4e64d307d8e8b20d568e782ece3926981d0a96c",
          "0x14a6c152ace4b16a42e1377e400798e15ac60320c21d90277890dbdd551d1912",
          "0x035992598be4d2ae5334f24f30355a1fca7f0e28762a178d79e993dca70eaabf",
          "0x03b645319eb70d7a692ea8c87fbcab1f292cd502071d9ffb7291ebe28356aa07",
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
          "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
          "0x13b360d4e82fe915fed16081038f98c211427b87a281bd733c277dbadf10372b",
          "0x08e7cbfea108224b0777f0558503af41585b75ab8d4d807505158f4bc8c771de",
          "0x0b6f861977ce57ddb2e647048ed9b9433cac640ba0599e264e24446765d915d3",
          "0x26f93d99832c6285d9abf8c5b896ea753ed137516972c5ddcc464b56c488d600",
          "0x086398ace043cf0db4e99f2712f392ddbd9c7b9202b2fcadf26e9b145316df9f",
          "0x04765b5102d6627cfcc389436e96bbc9aa478dcb720b546c77063077a40910ce",
          "0x2e39c17d3e15071a9341cfcea233ad5f25a14dea28716de88ae369ac2c9944ac",
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
          "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
          "0x107aab49e65a67f9da9cd2abf78be38bd9dc1d5db39f81de36bcfa5b4b039043",
          "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
          "0x2290ee31c482cf92b79b1944db1c0147635e9004db8c3b9d13644bef31ec3bd3",
          "0x1d59376149b959ccbd157ac850893a6f07c2d99b3852513ab8d01be8e846a566",
          "0x2d8040c3a09c49698c53bfcb514d55a5b39e9b17cb093d128b8783adb8cbd723",
          "0x231e38741f5c3f0ce2509bd3289b09a893b14b136eea6ed00cec9ac9271c9db5",
          "0x034183d253b6b250dae0a457797029523434d2b6bc41c09b6ef409bd970e4208",
          "0x1cb0ed9df901b713b97ee5357df1bf9b16f16cc0d737b31e07ba77d910efc8d6",
          "0x277c827440297ddeb0d85560fc7da91bcfd8729cec6bf01c3ecc664827388e23",
          "0x24f4c8596963484c0569feb1f2a79f19eb87843cd95fd26af5bdb12c8a26ea2e",
          "0x096b10d95e053da3dea44cf0c8ea6de7e962b0f71046aab3779baa3d2b772a01",
          "0x2800b5c600edd11c0366a68f6e8dc57f6a976cb6770673e351735a7f9ce92062",
          "0x2bedf321de5b3dacbb8cbc7312ea9c937dec5a7d07ae1c24ccdbc51c4bf6ef33",
          "0x022a8cf5a31c990f250e75e7df4daafe02929a5e514802a8b8fe8be7c366bb55",
          "0x06272e83847dd219527e22d89b87a6efdf7fc2b0f32d564762853d937378875d",
      },
      {
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x09226b6e22c6f0ca64ec26aad4c86e715b5f898e5e963f25870e56bbe533e9a2",
          "0x2a3d1fef5cb3ce1065d222dde5f9b16d48b42597868c1a49e15cb8007c8778a4",
          "0x0f34fda7cd268bf095354ea6c2631826c349d7518bf094361bc91f61c519c7fb",
          "0x2f549305063b1803a77ba92486c5383e00468fc857ec66d944868c4a309e596a",
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
          "0x112272f91e39f0f3754b36d52fafaa32b73194639e9077aeeb8fd9eeb5c197f6",
          "0x2c5b7efa2ecceeea67640c3a314bf22430a3635c16e265bf41b69cae74bb6ca8",
          "0x2b63732376158eeb59affea24c42fe2b9f5f0ed91512e866a5010c4a422a2210",
          "0x15ef7f2ee3cb842ed2df84652b6d941208ee251aa1d3d440fdef29ab9c91123d",
          "0x1efdce0e537a38a6167bb190151b64ce84ad66c093e64acda84995afbad32c38",
          "0x15fccfcbfbb7d19b3579da7e187ea1bfc00c071f5fcc688c162006b8fb3802c1",
          "0x044d3c43a1d8bdf98d9ce3973799577b3cc148f71f05c1381c7b28a42b010bdb",
          "0x2601b44a9ceaf52a979de420fd3c6a7d8ef043cd6b845368cf317809c7603844",
          "0x2238916f0b1084c82c027898e274eb46c7d8ac0c96a2b245d563706526ea705e",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x2c8157f94657d88c59af021da365441045bc63e1b38b4487c91b519f9ff62533",
          "0x27fb1c055aaf8a4205acd5be9a2be2e6718de3cb2c83c8e64a3dc0676bb8a945",
          "0x14c116fa641eb6536a6e13a3830fb449d47d31606dd888bdd65630ee6be537d8",
          "0x27edcb6842c33cd5a3127fa5ad1ad538ba7001c66e8b349b320ce35a0d11b8c1",
          "0x22cf5fc5ab1490168dd5c1acb0870ec4a605e9ca6353518e5012c3677d12dfab",
          "0x17e8e6e9a190194640ee7602c85d0c7aeb8bc51862d349be78fb720940e9833e",
      },
      {
          "0x1f09a0f7ba26bdc402f8617328d38bce213bad6def415a22398d2727cbfda3c6",
          "0x27918de28986445f3b86531a98ae3c84a22efadff737cb6654200067594f146b",
          "0x0e4adbd349654830f53e209fe0803dc6c9b38c1bd257ffeb4719661b05bca7c8",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x05ee0abc9f39a24a6c0448af3b29cc5b70b02961d28a7c75c4bf5665b6e33de3",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x089d2cd7436de0d9bdd87214d79aebde9f5f670221c1f5bc7a6905fa720e17ff",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x2dc0c43913c4431c08ed1d329bd5dddd857f8aec88417ee833a16c5814c27120",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x2ff80704d0f96302a799aa96eeff44148da396e5b3114bb690e7167eacf7da6f",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x24b74ff3c01e251b3a5e0ffe31b8ddb64b8fd2879fd76451063d87485fac338f",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
          "0x2766720e645263aa8c323963ce29fd397a3f1027ef0edd97bbe736dd1ad70dab",
          "0x07ba99b0454b12d883a90b7d9fc585a570a5609c117948673d653b75b4f8dbeb",
      },
      {
          "0x19e45f96ed87e5016568eaf3d9ddfa979cde528c9a9f889ea6bca69314ae07a8",
          "0x113ad62e5f1760833e3c623377b1da8c53b6df849f9d26ebb0dff4c13d89c129",
          "0x06a1a39fdf7310f4c8bd3dc6ba194a8c2c064994f519c4ad0bab55e6044ec707",
          "0x14ac062955c7bf10fba42f8d8917de76b0bf5acda89874fe70a323ce68dd67a7",
          "0x0608fe05ef99aa7bfd9e1d5c20d096da7e14e4bbf4c3757ecfb29aa624682461",
          "0x1e4d544947af4acf44e5a2f787700883bcf361afc726f707c15ecd73bf2991a2",
          "0x0c8b10d3d236bf6bb5d7855a3610b44707a62b6bf4cc12c5c7ea30fa87afe7ce",
          "0x0da3a967ee55218c7aeb0e79471780de9df3e0e43b99ab84492e5cee6178812c",
          "0x1f49ce3e263fe94985bc9c4fc8a6da078e88183b84f2d5eccbdd23322ed0e52a",
          "0x2f854e8576b393f8281a4ee34765fae491d456b5ebc255984aee6d1a6bfddb24",
          "0x251d65931d09aea2d18ce4339b3df0c44d97bcb6c89c5352b7033b959b751b4e",
          "0x126bc9da37891c2fb865fc63042ed2d0eac5970369a4c64b02e5c21180fb9905",
          "0x2caedffb118383bf855a6d2133dac4c209be168701ea35614d2972d596102a16",
          "0x08fc6c0a8cc8cd105ef293dbc8d9ea934ba1bbf9b15c190d110056914fb59ce8",
          "0x10eefb3ba0dde80ebdbe59c08c3cc0acf8c8112a8c117c16ffbb599693eb8e42",
          "0x2c0d237a46bd438641ab51032ea47e9416eefddba59bfa54cb0ca85994ae7e2a",
      },
      {
          "0x16b8295dfc7aac9a92f40830d0a18021db1c3c32cb0186b2bbea437e4992bc9e",
          "0x060b53de43d5419e8776eaefc4651c57b587045d5ac2ae857fe87be466e6c9d4",
          "0x2ac49b1b54eb7dffed2e32d42b9b780c717802960ace5b4be230326426266058",
          "0x26370289d5f64795d58f0d5cf98a2e694311753f5c5f0fc453d79600c3771454",
          "0x05b3d1abef33209714601dd3dfd3067cdf6aa1e9122d8e22a05ea1bcffbb9c6c",
          "0x300297d66a50a1974cc152a3dc9dc5cb9337b131b3255c36cdada2d141ac49f4",
          "0x19eacc03d9e4d435d98cc9717e74a7c39011da0e8780d9e26c7ac11e1d45d05e",
          "0x0f1d418da7ee6f503af23a71e6423b0dd1fb15de160753b2b961c42fd97b34e4",
          "0x094fff48f42c489724b45af5ff0bf5c125c8a524c2d1fa34029ee95313b6426c",
          "0x2b460fee9892bd0cf8c38333135a007b59a73bb46cccc65df68c525bc3ff4704",
          "0x25f8c1e4a021048da4b79acff3497b746fc72cf27c7351128d8db0c38d0dfeff",
          "0x142b65b35f124182a2c1bc038acbdf94da71b68ed1af1b1f0788ff44f44f7d4c",
          "0x10956ff3bf348daac08be5b154b80d5f5c5a6ab649135c5881905f83a42f14f7",
          "0x1e119c3a2345365cdf28ad7c4801925298d3b68aa4669027b1c0c6302fd293f4",
          "0x2c71de2892808d0391c25b0369e19f351bc0b4f0cdc4c6438c0374f410b980ba",
          "0x1d360eeea3494b3491a1a8901b219910755f94931eb85d5d0f97ca305ff1e9ea",
      },
  };

  constexpr static uint8_t kProof[] = {
      144, 229, 155, 204, 124, 168, 31,  49,  47,  46,  92,  241, 59,  42,  185,
      207, 170, 63,  142, 222, 11,  96,  209, 55,  68,  158, 132, 196, 160, 37,
      58,  9,   11,  179, 5,   174, 11,  70,  229, 110, 251, 12,  31,  8,   37,
      111, 214, 202, 175, 178, 120, 225, 0,   82,  92,  48,  127, 152, 74,  122,
      163, 9,   254, 153, 49,  52,  6,   115, 102, 78,  171, 187, 144, 238, 90,
      52,  145, 83,  130, 239, 132, 191, 0,   148, 105, 169, 113, 24,  122, 67,
      220, 66,  79,  109, 109, 145, 144, 229, 155, 204, 124, 168, 31,  49,  47,
      46,  92,  241, 59,  42,  185, 207, 170, 63,  142, 222, 11,  96,  209, 55,
      68,  158, 132, 196, 160, 37,  58,  9,   11,  179, 5,   174, 11,  70,  229,
      110, 251, 12,  31,  8,   37,  111, 214, 202, 175, 178, 120, 225, 0,   82,
      92,  48,  127, 152, 74,  122, 163, 9,   254, 153, 49,  52,  6,   115, 102,
      78,  171, 187, 144, 238, 90,  52,  145, 83,  130, 239, 132, 191, 0,   148,
      105, 169, 113, 24,  122, 67,  220, 66,  79,  109, 109, 145, 66,  195, 64,
      156, 233, 238, 146, 91,  109, 201, 118, 74,  47,  107, 32,  90,  228, 96,
      159, 192, 134, 170, 173, 118, 182, 145, 155, 120, 220, 10,  172, 10,  250,
      167, 203, 38,  247, 67,  71,  194, 136, 38,  90,  46,  88,  16,  89,  252,
      97,  209, 153, 172, 62,  193, 110, 232, 107, 144, 43,  234, 29,  23,  186,
      171, 18,  80,  164, 62,  2,   173, 176, 165, 24,  63,  198, 148, 44,  181,
      5,   25,  241, 196, 231, 76,  144, 49,  186, 149, 22,  252, 115, 106, 148,
      16,  186, 161, 3,   27,  163, 12,  185, 103, 196, 207, 58,  14,  233, 187,
      160, 165, 229, 43,  214, 100, 121, 38,  159, 72,  240, 57,  250, 190, 138,
      206, 126, 32,  232, 42,  72,  247, 125, 133, 48,  203, 17,  11,  15,  159,
      69,  60,  196, 211, 194, 92,  161, 43,  117, 173, 161, 247, 115, 9,   45,
      110, 189, 153, 173, 57,  74,  20,  200, 55,  85,  187, 107, 252, 50,  220,
      137, 36,  250, 216, 107, 213, 167, 153, 13,  17,  237, 150, 215, 84,  114,
      120, 113, 76,  118, 250, 155, 93,  177, 133, 201, 40,  195, 184, 255, 218,
      118, 222, 145, 112, 24,  249, 63,  132, 230, 84,  148, 76,  235, 216, 185,
      220, 226, 42,  77,  49,  218, 208, 226, 66,  24,  16,  138, 206, 72,  88,
      133, 139, 229, 177, 148, 1,   92,  47,  205, 198, 167, 227, 7,   38,  170,
      146, 208, 17,  155, 207, 249, 174, 33,  70,  200, 31,  200, 5,   1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      51,  55,  111, 142, 4,   28,  138, 39,  192, 209, 222, 33,  47,  75,  4,
      142, 178, 204, 244, 217, 201, 219, 137, 244, 240, 139, 135, 251, 199, 140,
      210, 137, 85,  252, 254, 173, 127, 30,  188, 222, 28,  239, 206, 132, 243,
      231, 98,  251, 98,  103, 69,  218, 106, 51,  156, 209, 231, 200, 53,  221,
      126, 97,  226, 23,  189, 206, 145, 190, 213, 66,  153, 91,  21,  223, 200,
      69,  86,  27,  100, 23,  45,  249, 45,  4,   224, 2,   26,  8,   149, 178,
      47,  28,  67,  107, 240, 13,  236, 101, 69,  224, 54,  73,  61,  136, 79,
      41,  179, 249, 174, 224, 239, 203, 2,   117, 165, 193, 79,  104, 156, 104,
      246, 14,  215, 120, 77,  133, 81,  32,  34,  45,  224, 187, 102, 150, 226,
      46,  18,  8,   84,  251, 83,  249, 132, 26,  243, 225, 109, 95,  96,  118,
      45,  251, 122, 208, 73,  42,  148, 107, 235, 39,  189, 206, 145, 190, 213,
      66,  153, 91,  21,  223, 200, 69,  86,  27,  100, 23,  45,  249, 45,  4,
      224, 2,   26,  8,   149, 178, 47,  28,  67,  107, 240, 13,  236, 101, 69,
      224, 54,  73,  61,  136, 79,  41,  179, 249, 174, 224, 239, 203, 2,   117,
      165, 193, 79,  104, 156, 104, 246, 14,  215, 120, 77,  133, 81,  32,  34,
      45,  224, 187, 102, 150, 226, 46,  18,  8,   84,  251, 83,  249, 132, 26,
      243, 225, 109, 95,  96,  118, 45,  251, 122, 208, 73,  42,  148, 107, 235,
      39,  98,  19,  202, 176, 9,   197, 247, 128, 131, 56,  212, 119, 122, 14,
      41,  254, 38,  112, 63,  208, 251, 12,  34,  55,  82,  160, 155, 195, 2,
      189, 227, 45,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   42,  61,  253, 254, 60,  108, 175, 204, 50,  55,
      157, 131, 169, 131, 97,  32,  47,  230, 8,   159, 254, 154, 195, 102, 36,
      89,  40,  145, 72,  150, 66,  33,  222, 68,  198, 34,  170, 47,  228, 103,
      141, 201, 105, 11,  137, 102, 0,   84,  106, 51,  228, 143, 17,  250, 60,
      248, 194, 231, 38,  132, 10,  22,  157, 45,  49,  132, 182, 46,  111, 96,
      142, 237, 36,  138, 123, 242, 134, 16,  110, 98,  46,  188, 66,  177, 101,
      1,   123, 251, 13,  168, 106, 183, 230, 204, 184, 45,  83,  246, 59,  197,
      101, 154, 27,  75,  77,  129, 127, 123, 13,  244, 201, 157, 82,  121, 86,
      14,  54,  181, 106, 64,  117, 121, 141, 203, 116, 97,  33,  17,  150, 108,
      76,  52,  78,  139, 209, 9,   104, 70,  36,  23,  128, 198, 100, 238, 253,
      219, 191, 93,  221, 65,  47,  50,  30,  69,  57,  35,  209, 160, 41,  10,
      240, 118, 141, 58,  171, 186, 248, 169, 198, 127, 183, 126, 206, 51,  119,
      184, 9,   228, 19,  40,  133, 191, 182, 87,  141, 23,  236, 71,  102, 208,
      87,  9,   50,  79,  238, 240, 103, 115, 77,  185, 254, 216, 215, 174, 129,
      47,  103, 85,  190, 68,  100, 251, 3,   99,  232, 254, 11,  195, 66,  227,
      12,  40,  90,  16,  85,  216, 182, 190, 227, 130, 69,  54,  71,  247, 32,
      162, 8,   49,  166, 175, 231, 66,  230, 248, 48,  187, 106, 152, 32,  195,
      206, 40,  134, 38,  180, 20,  120, 236, 78,  85,  145, 46,  245, 22,  230,
      101, 194, 96,  45,  6,   0,   211, 131, 100, 68,  78,  161, 65,  161, 44,
      92,  160, 254, 106, 152, 169, 164, 15,  117, 35,  75,  249, 230, 98,  230,
      3,   101, 97,  122, 230, 70,  254, 208, 166, 172, 97,  140, 141, 128, 86,
      178, 203, 39,  219, 101, 144, 187, 198, 6,   26,  226, 6,   198, 229, 69,
      245, 78,  131, 98,  105, 4,   225, 83,  52,  246, 72,  51,  50,  92,  132,
      12,  101, 46,  76,  18,  5,   32,  238, 253, 29,  5,   12,  21,  54,  99,
      83,  141, 46,  210, 247, 240, 159, 93,  231, 243, 252, 10,  235, 142, 148,
      247, 77,  3,   246, 235, 236, 205, 75,  195, 32,  210, 49,  200, 10,  196,
      60,  60,  117, 125, 13,  15,  91,  25,  164, 70,  214, 135, 44,  82,  110,
      89,  103, 204, 49,  208, 64,  176, 90,  97,  148, 110, 181, 93,  136, 15,
      1,   19,  132, 211, 138, 98,  98,  23,  228, 141, 107, 202, 34,  238, 136,
      96,  36,  202, 128, 233, 104, 112, 46,  75,  198, 127, 245, 107, 167, 51,
      90,  102, 23,  70,  122, 156, 119, 158, 161, 141, 128, 3,   108, 34,  117,
      36,  13,  40,  204, 245, 155, 179, 128, 31,  45,  252, 196, 177, 217, 58,
      45,  202, 98,  211, 22,  250, 124, 29,  10,  175, 40,  234, 81,  80,  108,
      44,  198, 93,  252, 23,  45,  162, 254, 10,  246, 100, 107, 60,  76,  141,
      169, 250, 150, 191, 37,  251, 46,  216, 8,   50,  15,  170, 187, 165, 33,
      154, 209, 157, 172, 235, 192, 17,  148, 105, 245, 225, 140, 18,  245, 45,
      207, 150, 250, 23,  124, 92,  182, 134, 44,  242, 128, 66,  181, 202, 241,
      159, 30,  52,  212, 139, 225, 247, 127, 16,  230, 7,   51,  111, 54,  200,
      239, 132, 148, 131, 21,  51,  179, 63,  161, 134, 27,  186, 83,  246, 123,
      47,  234, 184, 199, 37,  51,  61,  243, 10,  19,  109, 64,  46,  197, 9,
      29,  45,  90,  227, 97,  192, 225, 255, 158, 95,  152, 120, 26,  179, 223,
      141, 22,  141, 32,  29,  254, 46,  202, 127, 195, 14,  143, 53,  243, 174,
      155, 90,  18,  7,   69,  171, 59,  28,  172, 202, 190, 166, 62,  132, 15,
      254, 83,  165, 161, 171, 214, 106, 13,  144, 191, 209, 56,  42,  97,  231,
      97,  96,  255, 115, 167, 170, 130, 182, 216, 29,  222, 219, 199, 85,  76,
      95,  45,  46,  151, 117, 253, 18,  106, 95,  98,  223, 189, 149, 25,  140,
      241, 182, 165, 133, 234, 187, 53,  137, 189, 203, 35,  0,   248, 9,   174,
      125, 27,  20,  32,  16,  54,  52,  179, 204, 160, 254, 238, 231, 85,  248,
      244, 159, 243, 12,  178, 40,  5,   238, 41,  215, 95,  165, 143, 180, 5,
      35,  189, 244, 151, 209, 38,  45,  35,  62,  100, 75,  3,   128, 229, 94,
      120, 206, 100, 21,  97,  192, 204, 221, 10,  234, 1,   12,  138, 234, 97,
      153, 6,   89,  240, 152, 239, 75,  13,  5,   245, 65,  199, 236, 52,  135,
      114, 155, 224, 201, 186, 242, 20,  91,  5,   206, 175, 163, 88,  240, 134,
      8,   2,   209, 106, 216, 1,   33,  229, 105, 45,  3,   149, 247, 73,  142,
      64,  229, 157, 63,  227, 49,  227, 185, 188, 156, 240, 212, 16,  233, 168,
      209, 144, 162, 90,  230, 120, 157, 27,  114, 88,  33,  6,   95,  104, 167,
      173, 103, 30,  122, 2,   254, 75,  189, 228, 79,  160, 203, 84,  52,  7,
      153, 0,   155, 189, 24,  2,   31,  86,  220, 207, 204, 164, 73,  138, 6,
      100, 140, 233, 37,  146, 104, 88,  207, 55,  43,  171, 68,  137, 70,  122,
      187, 198, 95,  200, 255, 5,   47,  85,  245, 2,   141, 153, 160, 236, 147,
      4};

  // clang-format off
  constexpr static Point kAdviceCommitments[][3] = {
    {
      {"0x093a25a0c4849e4437d1600bde8e3faacfb92a3bf15c2e2f311fa87ccc9be590",
       "0x23f368976bd1e762f05c5efc8db4e7590d2300e60ef4a7bd898cc2f5bbd771e6"},
      {"0x19fe09a37a4a987f305c5200e178b2afcad66f25081f0cfb6ee5460bae05b30b",
       "0x0e106dd8def54c87c517f1eb20a56175822ce6e65f99a3a0655fc3e2e8a3e94b"},
      {"0x116d6d4f42dc437a1871a9699400bf84ef825391345aee90bbab4e6673063431",
       "0x2c2d516d5c04579cca6c30460e27130515680a99050770aabcdd8ec5855a9373"},
    },
    {
      {"0x093a25a0c4849e4437d1600bde8e3faacfb92a3bf15c2e2f311fa87ccc9be590",
       "0x23f368976bd1e762f05c5efc8db4e7590d2300e60ef4a7bd898cc2f5bbd771e6"},
      {"0x19fe09a37a4a987f305c5200e178b2afcad66f25081f0cfb6ee5460bae05b30b",
       "0x0e106dd8def54c87c517f1eb20a56175822ce6e65f99a3a0655fc3e2e8a3e94b"},
      {"0x116d6d4f42dc437a1871a9699400bf84ef825391345aee90bbab4e6673063431",
       "0x2c2d516d5c04579cca6c30460e27130515680a99050770aabcdd8ec5855a9373"},
    },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x0e147bc018ea0dae60978bd4c802f89281f85d9cefaeef97e8b1c59d058825ef";

  constexpr static std::string_view kBeta =
      "0x00e8932e10163041ad7c218ae33a024e9bd27598bc694a390cb45feacf635f03";

  constexpr static std::string_view kGamma =
      "0x293b80e793dafce1f1db5e5d7f266c9f18f22a29e2d3cac7437c0eb02964d584";

  // clang-format off
  constexpr static Point kPermutationProductCommitments[][4] = {
      {
          {"0x0aac0adc789b91b676adaa86c09f60e45a206b2f4a76c96d5b92eee99c40c342",
            "0x02d89be67f1f0a190db5ac710351156edd14bd4c1c8d971e0c189b24cc1fd25c"},
          {"0x2bba171dea2b906be86ec13eac99d161fc5910582e5a2688c24743f726cba7fa",
            "0x2a83885d5e5d33ce854487c7b3c4f660ea1a0397d506b3309e9d3725ab222a55"},
          {"0x21ba10946a73fc1695ba31904ce7c4f11905b52c94c63f18a5b0ad023ea45012",
            "0x046efe02c21b19217f8bc9347dcd14a45b7fa8a637e140c158db26c2a666feed"},
          {"0x2ae8207ece8abefa39f0489f267964d62be5a5a0bbe90e3acfc467b90ca31b03",
            "0x069f5af770fd98d51ab2d9870ad1320a849edd453e5018b423ee66592157b0da"},
      },
      {
          {"0x144a39ad99bd6e2d0973f7a1ad752ba15cc2d3c43c459f0f0b11cb30857df748",
            "0x0de06616abf1d127658e5abb001513408168702f80581e12d817d0f5dcb3a9c0"},
          {"0x05b15d9bfa764c71787254d796ed110d99a7d56bd8fa2489dc32fc6bbb5537c8",
            "0x2463a4ac48f4955fa0aa849200e950d14f85616ffffb51bf9af080d68f676b31"},
          {"0x101842e2d0da314d2ae2dcb9d8eb4c9454e6843ff9187091de76daffb8c328c9",
            "0x11356be411434aed21ed05ef34e4dd688a3794f311f35979c033cb55cb6bd8bc"},
          {"0x05c81fc84621aef9cf9b11d092aa2607e3a7c6cd2f5c0194b1e58b855848ce8a",
            "0x12466a17eb84900aa737801b9adf80ca99482e3e1045a78df5bc9777747e1710"},
      },
  };
  // clang-format on

  constexpr static std::string_view kY =
      "0x287bf794a84ee4b639e643202679380ef8cab5bfd8b3b6b30df1645e170e79f5";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x09d28cc7fb878bf0f489dbc9d9f4ccb28e044b2f21ded1c0278a1c048e6f3733",
       "0x23612388ba8985a9ac2a94e8c79ae5d6ac1d97e5e48582d0c49625fe450766e5"},
      {"0x17e2617edd35c8e7d19c336ada456762fb62e7f384ceef1cdebc1e7fadfefc55",
       "0x061f2154c9097a1ad8f784c6bf043e59142944506ccddc2bbc0fc7c01c50ff36"},
  };

  constexpr static std::string_view kX =
      "0x2d91afabb171fe83cb29e3753f33a39afc4bbe033bc15d77b776d2b00ae85823";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x0df06b431c2fb295081a02e0042df92d17641b5645c8df155b9942d5be91cebd",
          "0x2051854d78d70ef6689c684fc1a57502cbefe0aef9b3294f883d4936e04565ec",
          "0x27eb6b942a49d07afb2d76605f6de1f31a84f953fb5408122ee29666bbe02d22",
      },
      {
          "0x0df06b431c2fb295081a02e0042df92d17641b5645c8df155b9942d5be91cebd",
          "0x2051854d78d70ef6689c684fc1a57502cbefe0aef9b3294f883d4936e04565ec",
          "0x27eb6b942a49d07afb2d76605f6de1f31a84f953fb5408122ee29666bbe02d22",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x2de3bd02c39ba05237220cfbd03f7026fe290e7a77d4388380f7c509b0ca1362",
  };

  constexpr static std::string_view kCommonPermutationEvals[] = {
      "0x214296489128592466c39afe9f08e62f206183a9839d3732ccaf6c3cfefd3d2a",
      "0x2d9d160a8426e7c2f83cfa118fe4336a540066890b69c98d67e42faa22c644de",
      "0x2db8cce6b76aa80dfb7b0165b142bc2e626e1086f27b8a24ed8e606f2eb68431",
      "0x11216174cb8d7975406ab5360e5679529dc9f40d7b7f814d4b1b9a65c53bf653",
  };

  constexpr static std::string_view kPermutationProductEvals[][4] = {
      {
          "0x0a29a0d12339451e322f41dd5dbfdbfdee64c6801724466809d18b4e344c6c96",
          "0x14b4268628cec320986abb30f8e642e7afa63108a220f747364582e3beb6d855",
          "0x0c051dfdee2005124c2e650c845c323348f63453e1046962834ef545e5c606e2",
          "0x17665a33a76bf57fc64b2e7068e980ca246088ee22ca6b8de41762628ad38413",
      },
      {
          "0x2efb25bf96faa98d4c3c6b64f60afea22d17fc5dc62c6c5051ea28af0a1d7cfa",
          "0x1a78985f9effe1c061e35a2d1d09c52e406d130af33d3325c7b8ea2f7bf653ba",
          "0x20141b7dae09f80023cbbd8935bbea85a5b6f18c1995bddf625f6a12fd75972e",
          "0x2d69e52101d86ad1020886f058a3afce055b14f2bac9e09b728734ecc741f505",
      },
  };

  constexpr static std::string_view kPermutationProductNextEvals[][4] = {
      {
          "0x0957d06647ec178d57b6bf852813e409b87733ce7eb77fc6a9f8baab3a8d76f0",
          "0x0fa4a9986afea05c2ca141a14e446483d300062d60c265e616f52e91554eec78",
          "0x0ac831d220c34bcdecebf6034df7948eeb0afcf3e75d9ff0f7d22e8d53633615",
          "0x16d362ca2d3ad9b1c4fc2d1f80b39bf5cc280d2475226c03808da19e779c7a46",
      },
      {
          "0x2c86b65c7c17fa96cf2df5128ce1f5699411c0ebac9dd19a21a5bbaa0f3208d8",
          "0x0f843ea6becaac1c3bab4507125a9baef3358f0ec37fca2efe1d208d168ddfb3",
          "0x26d197f4bd2305b48fa55fd729ee0528b20cf39ff4f855e7eefea0ccb3343610",
          "0x062158721b9d78e65aa290d1a8e910d4f09cbcb9e331e33f9de5408e49f79503",
      },
  };

  constexpr static std::string_view kPermutationProductLastEvals[][4] = {
      {
          "0x105a280ce342c30bfee86303fb6444be55672f81aed7d8feb94d7367f0ee4f32",
          "0x1a06c6bb9065db27cbb256808d8c61aca6d0fe46e67a616503e662e6f94b2375",
          "0x010f885db56e94615ab040d031cc67596e522c87d646a4195b0f0d7d753c3cc4",
          "",
      },
      {
          "0x1b86a13fb33315839484efc8366f3307e6107ff7e18bd4341e9ff1cab54280f2",
          "0x2d5f4c55c7dbde1dd8b682aaa773ff6061e7612a38d1bf900d6ad6aba1a553fe",
          "0x0d4bef98f059069961ea8a0c01ea0addccc0611564ce785ee580034b643e232d",
          "",
      },
  };

  static void TestConfig(Fibonacci1Config<F>& config) {
    std::array<AdviceColumnKey, 3> expected_advice = {
        AdviceColumnKey(0),
        AdviceColumnKey(1),
        AdviceColumnKey(2),
    };
    EXPECT_EQ(config.advice, expected_advice);
    EXPECT_EQ(config.instance, InstanceColumnKey(0));
    EXPECT_EQ(config.selector, Selector::Simple(0));
  }

  static std::vector<Evals> GetInstanceColumns() {
    F a(1);
    F b(1);
    F out(55);
    std::vector<F> instance_column = {std::move(a), std::move(b),
                                      std::move(out)};
    return std::vector<Evals>{Evals(std::move(instance_column))};
  }
};

// V1FloorPlanner
template <typename Circuit, typename PCS, typename LS>
class Fibonacci1TestData<Circuit, PCS, LS,
                         std::enable_if_t<IsV1FloorPlanner<Circuit>>>
    : public CircuitTestData<Circuit, PCS, LS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;

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
        "num_fixed_columns: 0, "
        "num_advice_columns: 3, "
        "num_instance_columns: 1, "
        "num_selectors: 1, "
        "gates: [Product("
          "Selector(Selector(0, true)), "
          "Sum("
            "Sum("
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
              "column_index: 2, "
              "rotation: Rotation(0) "
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
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Rotation(0)"
          ")"
        "], "
        "instance_queries: [("
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}, "
          "Rotation(0)"
        ")], "
        "fixed_queries: [], "
        "permutation: Argument { columns: ["
          "Column { "
            "index: 0, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 1, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 2, "
            "column_type: Advice "
          "}, "
          "Column { "
            "index: 0, "
            "column_type: Instance "
          "}"
        "] }, "
        "lookups: [], "
        "constants: [], "
        "minimum_degree: None "
      "}";
  // clang-format on

  constexpr static AnyColumnKey kAssemblyPermutationColumns[] = {
      AdviceColumnKey(0),
      AdviceColumnKey(1),
      AdviceColumnKey(2),
      InstanceColumnKey(0),
  };

  // clang-format off
  constexpr static Label kCycleStoreMapping[][kN] = {
      {{1, 1}, {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {3, 1},  {3, 0},
        {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 1}, {2, 2},  {2, 3},  {2, 4},  {2, 5},  {2, 6},  {2, 7},  {0, 6},
        {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{3, 2}, {1, 0},  {0, 0},  {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},
        {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 7}, {1, 7},  {2, 0},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
        {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };

  constexpr static Label kCycleStoreAux[][kN] = {
      {{1, 1}, {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},  {0, 7},
        {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{1, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
        {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 0}, {1, 0},  {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},
        {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 7}, {1, 7},  {2, 0},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
        {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  };
  // clang-format on

  constexpr static size_t kCycleStoreSizes[][kN] = {
      {1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1},
      {2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
      {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  };

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true,  true,  true,  true,  true,  true,  true,  true,
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
          "num_fixed_columns: 1, "
          "num_advice_columns: 3, "
          "num_instance_columns: 1, "
          "num_selectors: 1, "
          "gates: [Product("
            "Fixed { "
              "query_index: 0, "
              "column_index: 0, "
              "rotation: Rotation(0) "
            "}, "
            "Sum("
              "Sum("
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
                "column_index: 2, "
                "rotation: Rotation(0) "
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
                "index: 2, "
                "column_type: Advice "
              "}, "
              "Rotation(0)"
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
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 1, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 2, "
              "column_type: Advice "
            "}, "
            "Column { "
              "index: 0, "
              "column_type: Instance "
            "}"
          "] }, "
          "lookups: [], "
          "constants: [], "
          "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
          "(0x297ff8a661d1fa1196c065b6fb7df901fb16b5b83168ddca6c7749d963cc967a, "
            "0x1a7b1f2b5f3e35fc4c706ece6b8f646e95904f9aa417f8a31fd37a41da167ec1)"
        "], "
        "permutation: VerifyingKey { commitments: ["
          "(0x0d753ca469cfea858cfa6ec91acd016399cd294f61ff0d43ebb968d2524a713c, "
            "0x2550d6fb1e5c470ba3e6017989a6d380a4c0c6be14ba6e4275dae09e16a23284), "
          "(0x1357bd4a7abf61580c4d6830f369d242e2268747198bfe5e41cc960f5c475f77, "
            "0x29ca42708ce07d2576298db193d3fd9bf87739177d4bf2b831841798d8b2b50d), "
          "(0x1eba9015d88cec13d423a6c37c5edd453a72e087fcfb36461bf75dd35ebe344a, "
            "0x22c70d284caba78df7da1a1eb8607e5fa298a4e8ff21b8bfeb7b0772f4cb2c71), "
          "(0x1dd1199d21ff272e331fca7683198fed95caea81dd571af67753b96d29111b1d, "
            "0x0274945484229b0f6dcff00e591c859ac06bdb68ccd0bf25425ccdc266df1329)"
        "] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x0cf64676a870bcf55c113fb62e896e924807935cb63a2794b382539e995f8673";

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
  };

  constexpr static std::string_view kPermutationsColumns[][kN] = {
      {
          "0x133f51f46c2a51201fbb89dc1d6868acfef0f7ce0e572ccb98e04e99eac3ea36",
          "0x1240a374ee6f71e12df10a129946ca9ba824f58a50e3a58b4c68dd5590b74ad8",
          "0x009553089042fe83ab570b54e4e64d307d8e8b20d568e782ece3926981d0a96c",
          "0x14a6c152ace4b16a42e1377e400798e15ac60320c21d90277890dbdd551d1912",
          "0x035992598be4d2ae5334f24f30355a1fca7f0e28762a178d79e993dca70eaabf",
          "0x03b645319eb70d7a692ea8c87fbcab1f292cd502071d9ffb7291ebe28356aa07",
          "0x2f0e061e83e8b12c1bdf8df7cca02483295e89f78a462ce38859661e06501815",
          "0x18afdf23e9bd9302673fc1e076a492d4d65bd18ebc4d854ed189139bab313e52",
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
          "0x08e7cbfea108224b0777f0558503af41585b75ab8d4d807505158f4bc8c771de",
          "0x0b6f861977ce57ddb2e647048ed9b9433cac640ba0599e264e24446765d915d3",
          "0x26f93d99832c6285d9abf8c5b896ea753ed137516972c5ddcc464b56c488d600",
          "0x086398ace043cf0db4e99f2712f392ddbd9c7b9202b2fcadf26e9b145316df9f",
          "0x04765b5102d6627cfcc389436e96bbc9aa478dcb720b546c77063077a40910ce",
          "0x2e39c17d3e15071a9341cfcea233ad5f25a14dea28716de88ae369ac2c9944ac",
          "0x2a3d1fef5cb3ce1065d222dde5f9b16d48b42597868c1a49e15cb8007c8778a4",
          "0x1d59376149b959ccbd157ac850893a6f07c2d99b3852513ab8d01be8e846a566",
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
          "0x034183d253b6b250dae0a457797029523434d2b6bc41c09b6ef409bd970e4208",
          "0x09226b6e22c6f0ca64ec26aad4c86e715b5f898e5e963f25870e56bbe533e9a2",
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
          "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
          "0x107aab49e65a67f9da9cd2abf78be38bd9dc1d5db39f81de36bcfa5b4b039043",
          "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
          "0x2290ee31c482cf92b79b1944db1c0147635e9004db8c3b9d13644bef31ec3bd3",
          "0x1cb0ed9df901b713b97ee5357df1bf9b16f16cc0d737b31e07ba77d910efc8d6",
          "0x277c827440297ddeb0d85560fc7da91bcfd8729cec6bf01c3ecc664827388e23",
          "0x24f4c8596963484c0569feb1f2a79f19eb87843cd95fd26af5bdb12c8a26ea2e",
          "0x096b10d95e053da3dea44cf0c8ea6de7e962b0f71046aab3779baa3d2b772a01",
          "0x2800b5c600edd11c0366a68f6e8dc57f6a976cb6770673e351735a7f9ce92062",
          "0x2bedf321de5b3dacbb8cbc7312ea9c937dec5a7d07ae1c24ccdbc51c4bf6ef33",
          "0x022a8cf5a31c990f250e75e7df4daafe02929a5e514802a8b8fe8be7c366bb55",
          "0x06272e83847dd219527e22d89b87a6efdf7fc2b0f32d564762853d937378875d",
      },
      {
          "0x2d8040c3a09c49698c53bfcb514d55a5b39e9b17cb093d128b8783adb8cbd723",
          "0x231e38741f5c3f0ce2509bd3289b09a893b14b136eea6ed00cec9ac9271c9db5",
          "0x13b360d4e82fe915fed16081038f98c211427b87a281bd733c277dbadf10372b",
          "0x0f34fda7cd268bf095354ea6c2631826c349d7518bf094361bc91f61c519c7fb",
          "0x2f549305063b1803a77ba92486c5383e00468fc857ec66d944868c4a309e596a",
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
          "0x112272f91e39f0f3754b36d52fafaa32b73194639e9077aeeb8fd9eeb5c197f6",
          "0x0f6ca21bfbcb52a4aee917564277498d8adcd0ead26ba0df100a7286b752d9d4",
          "0x15a7e8359f35c766c29ed77f4d1d5119901d036e20c15142192a21e09041e827",
          "0x0b4bc964d379eeaa158eca75b6dbb6a7b86f896b10f25adc9b92d8d5475ed2f1",
          "0x25d115149ca21b94ee56a92ef0d8c8602163a397afc69182e9b4411789e2e60b",
          "0x1e4a361c58a1fc2e342bb1b5522c187cfc6134683e6fa0f85eacc9916574db2c",
          "0x18b8417cd7b83ee5b62a8acacf2af01e8ea1a740a58c55047d8c926b8368c14d",
          "0x102953096b3b2549bc2c5f8c310a262a6280534c7e1984c250448ba8b1ff20d1",
          "0x12deb8e349e34c7dbd7857f75ca3dc59a3258bfe82bc3d01f2c8be9df84e4b96",
          "0x14232df27fdebe635d88be19813451e8b574c2458d8ae3c1503dc0833e136502",
          "0x25333b49ba4fe6f17bfa2659b5254c7ab6805b8bb4711ee45163adcdeeafdded",
          "0x13519d493b4d53fdd4aa5e6bfbf50e5e459112c26c54b865e524adc12e3c211e",
          "0x28b5a8a4f3f4184b8747b9d14a086e6e83e878fb40ffc9529d9603cf20a5ce1b",
          "0x105c187bb3938b9863b8a65abb1e84eeff1390bb1f5b99a6f34e1b0f8286db1b",
          "0x1d47bb6d01831abc30218432bcd7f6ebdf7fcdfd9e1c3c42302bb8d9526ed60c",
          "0x019d94bb672f383811eabd489da66d2cc7b2f8ce7e58d88c72e5e354c865eb21",
      },
      {
          "0x141e39f89c516db1255ecee3825495dfb18b1bd87123d9257363ae894a291dd3",
          "0x08a4dfd16be05cd2087a54d987991410bba11d1548080afde146412d7de47930",
          "0x2f25daf535a35461b7c8d7f63bdbbefbd8339e1d3c294d9069a788bb901c525c",
          "0x0ec047746f9977d98dc53b31063cd1fe923a04dfa0e7b689138388a53ce7d701",
          "0x256defc2c8f2e2dad221ba4401452d4a6aa8dced60798b0cdc3618e76a4ee2e0",
          "0x22053aee890a5e747c94c48401983f3c2e9cf8aaf7c8c03afaa6c573ef6fa7f8",
          "0x15add4829e360738c4aa9bf9a1ada88926b9bd475b09577c34ac62ebefcea7f6",
          "0x028b81cb54900a1ac0a193c1d76a3a83742de52b5ceb991e475e98982a564c33",
          "0x0361f1609cfec0d84058dbc7fac6dc7f3dde7c674b560517b375a6775bc71d65",
          "0x00603be4f6072cb8adb7b6a6a54cfac89e3c6287ade9d4cf64cbb1d659193490",
          "0x1a13eebb730009c87bc2fb35f61a4f938809953b467d85a8d6c4db8fa2d8547d",
          "0x21a406fe719828502a8b0a857b44865e95f9e368d8d1ba08305e6ceeb3182900",
          "0x1fbfc06d752ce7940fe2ad597c54a9ff0974b915ce28b93da3d3a08a5c60c1e8",
          "0x0e5f1384582741b53bbb81327fe91920f996ef9d81f0b056493b302000905809",
          "0x05de7057059dbb0bf2947a408fca011bb4cf784db57515afb28f9cee22669553",
          "0x2dd8cca78ca1960ef7aeb1f4aa171dd9b406031d1ccdd772fc835cfbc5a9b3ce",
      },
      {
          "0x15a78e0e6085f1debac2f4ec73d13ceba739fee72ad97c78b52a98f544aeb8ff",
          "0x1812dc7c7a640429efb6c9f4517a6a26665d974c956fc4b205154578966cb7d5",
          "0x2d25b29b330ef94426a3147d0cfbeac4967b6ad86802c928957e0e84a21a5d87",
          "0x261f4d63c9fdebb6334bf99c16a2b16a6889dbe3db6d53c72bba22395b7b672f",
          "0x1e79eccbbbe54d40275ee2bfbec7ae64b7aebf6e5142e793b30a7ec56dde6aff",
          "0x2f0b5305526a51f04ed9e81f590d5dadf43a2488ebf665c4f85c9ae9edf419c9",
          "0x1a268af4032dced16da12867c46aa1da4b8cbd95767a0829326fe829756fdeda",
          "0x160198e6359a11c21826c7d41371a65a056720e7f3a4ef9036a118e182490fd4",
          "0x27f8f5e308e267a585e5f57c4f1d6a7221517bb415c7de49238e98966bc593b1",
          "0x0b78ca79f16a7d1dae622114e268f3c124081d6a39858811f478685428e2adfc",
          "0x163129eea9b2d7f90a9491c5b1b14501d8446f1f6c89d26f192946472f70d883",
          "0x1a716c12432e1e01a617216c047e29823135fc00fb4cfb292e18d574606f35c0",
          "0x2177d56ea2af7bf7e9b131d78074426092359e6ca2345b36d32c1d2f03566ac3",
          "0x11856670bac1b7c78a8932e8c2137d3ea585b35beac3e92b61765cc3cdf68326",
          "0x26df134ce46792b7f32f53ac7df54484a1da3922ab3cab65ceba02399b5ace1e",
          "0x2a8f208fd791f7f5c13c533407af34929458b6fce3155f602331decc39a18d1b",
      },
      {
          "0x25e061e5a751efd01b3390c7dd2d33bc4071216db8e513d573a5c9ba9b66913a",
          "0x2c70d8ce1e96bc83662ff9edf413c8b1ddb7308b30770b8b0363bb16b410966b",
          "0x0fdf631b3cbc1368612327cce61e043125889ca33418bf1b2795fa110bdf77bc",
          "0x233d0ce0f5d908c63505bfca1434d15fc248b560ea827e14ee07d9498ddc8588",
          "0x20741541c6dd548e731c08a36e05f26bcd37a4bf5bd8507b67748ff15b8eaa1e",
          "0x06ceaea47f22a3a3f3cb1fd1a920bad815977860c229146ce63786165b6bac01",
          "0x2516e1b23de252185b9b2c8753c52dad78073c4848830e0ad3b139d7e1043c7b",
          "0x24202f1572d431b51140eab5b4305fff78babdd7726b235a0e8360b97ff1780f",
          "0x275e5b9af3e37e1402656f0c71add7179326c94a3c508b97e3136325374d0086",
          "0x043a8f7d69c75e225994410c0bbbb1b0cdedc772dd29f8b02444d05ab3daaaa2",
          "0x2e7cc9c94b8356104b5c50ae98eec937238938d0b15fd25da84d80a7817a6bcb",
          "0x143def6354020635267fb1edc79f0f36da06d1904800c28b2be63348fd2f7606",
          "0x1c951d6176ed6399f36cb6b8ddece674c50fe1b2415ef397aaff428a29246a59",
          "0x0ab51b71f8d9ba71c7319ad8657f90942a9701c88b7d6f04a51216475f40f779",
          "0x26d4e18d12de6a7b7c071a2dc7fbe89ea4c65f6d647143fcc5984acae9088635",
          "0x2eacc4ca1c50e55c175b62c30ac9a31e50d52c60ed7e7f8640bb920f4c69329a",
      },
  };

  constexpr static uint8_t kProof[] = {
      197, 90,  54,  138, 254, 187, 187, 144, 27,  39,  20,  39,  15,  181, 249,
      52,  63,  84,  95,  51,  45,  242, 210, 62,  170, 158, 176, 93,  60,  111,
      37,  2,   93,  88,  146, 144, 43,  179, 6,   218, 194, 31,  174, 181, 184,
      44,  226, 22,  117, 37,  50,  100, 239, 5,   38,  36,  177, 245, 5,   62,
      67,  52,  53,  46,  84,  190, 108, 245, 192, 106, 51,  107, 85,  215, 124,
      33,  178, 186, 27,  105, 218, 50,  59,  250, 73,  229, 251, 140, 191, 110,
      167, 71,  86,  11,  127, 25,  197, 90,  54,  138, 254, 187, 187, 144, 27,
      39,  20,  39,  15,  181, 249, 52,  63,  84,  95,  51,  45,  242, 210, 62,
      170, 158, 176, 93,  60,  111, 37,  2,   93,  88,  146, 144, 43,  179, 6,
      218, 194, 31,  174, 181, 184, 44,  226, 22,  117, 37,  50,  100, 239, 5,
      38,  36,  177, 245, 5,   62,  67,  52,  53,  46,  84,  190, 108, 245, 192,
      106, 51,  107, 85,  215, 124, 33,  178, 186, 27,  105, 218, 50,  59,  250,
      73,  229, 251, 140, 191, 110, 167, 71,  86,  11,  127, 25,  15,  108, 79,
      231, 72,  84,  56,  117, 61,  40,  248, 185, 208, 110, 150, 13,  117, 88,
      226, 218, 102, 120, 33,  190, 111, 7,   135, 197, 183, 246, 172, 44,  22,
      228, 172, 219, 254, 237, 163, 0,   10,  251, 101, 190, 60,  235, 177, 50,
      81,  39,  245, 199, 147, 61,  207, 69,  242, 102, 103, 215, 193, 129, 74,
      161, 49,  31,  185, 84,  59,  60,  197, 105, 162, 2,   73,  46,  192, 23,
      112, 64,  184, 76,  138, 38,  135, 129, 7,   34,  254, 199, 199, 128, 203,
      133, 125, 4,   7,   193, 249, 119, 43,  96,  224, 186, 80,  229, 177, 168,
      68,  165, 160, 232, 250, 119, 40,  233, 224, 122, 147, 180, 215, 108, 101,
      132, 148, 172, 39,  145, 155, 96,  69,  102, 243, 137, 112, 78,  242, 80,
      91,  105, 121, 10,  50,  254, 172, 156, 161, 108, 106, 0,   49,  244, 77,
      227, 250, 2,   241, 50,  80,  14,  167, 174, 122, 194, 14,  106, 139, 84,
      233, 24,  202, 125, 123, 229, 196, 200, 149, 88,  110, 141, 209, 208, 32,
      20,  184, 161, 4,   183, 72,  66,  77,  174, 122, 92,  96,  251, 2,   148,
      175, 199, 140, 72,  172, 207, 191, 86,  237, 169, 161, 36,  54,  37,  120,
      68,  72,  75,  34,  131, 253, 136, 155, 230, 167, 36,  102, 191, 16,  3,
      178, 239, 166, 170, 93,  26,  16,  155, 212, 248, 241, 61,  87,  232, 92,
      226, 73,  81,  194, 54,  103, 175, 237, 175, 51,  8,   101, 141, 1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      184, 103, 65,  174, 109, 225, 89,  252, 161, 224, 217, 194, 126, 110, 105,
      75,  66,  152, 227, 135, 90,  133, 7,   10,  97,  165, 139, 202, 145, 161,
      151, 151, 42,  93,  124, 131, 213, 41,  224, 106, 205, 160, 84,  83,  192,
      164, 91,  92,  117, 227, 188, 172, 70,  79,  136, 95,  223, 99,  129, 209,
      11,  49,  89,  4,   229, 144, 159, 187, 187, 178, 215, 128, 29,  220, 161,
      34,  115, 90,  56,  8,   223, 250, 67,  35,  89,  196, 146, 101, 84,  251,
      228, 103, 168, 95,  157, 33,  185, 194, 131, 159, 28,  136, 166, 197, 30,
      41,  212, 117, 36,  14,  230, 230, 54,  29,  207, 135, 111, 5,   94,  61,
      70,  5,   110, 98,  119, 252, 82,  48,  33,  182, 224, 230, 183, 171, 84,
      17,  96,  170, 243, 125, 224, 61,  200, 1,   11,  62,  231, 238, 65,  68,
      71,  90,  24,  129, 195, 168, 209, 140, 242, 47,  229, 144, 159, 187, 187,
      178, 215, 128, 29,  220, 161, 34,  115, 90,  56,  8,   223, 250, 67,  35,
      89,  196, 146, 101, 84,  251, 228, 103, 168, 95,  157, 33,  185, 194, 131,
      159, 28,  136, 166, 197, 30,  41,  212, 117, 36,  14,  230, 230, 54,  29,
      207, 135, 111, 5,   94,  61,  70,  5,   110, 98,  119, 252, 82,  48,  33,
      182, 224, 230, 183, 171, 84,  17,  96,  170, 243, 125, 224, 61,  200, 1,
      11,  62,  231, 238, 65,  68,  71,  90,  24,  129, 195, 168, 209, 140, 242,
      47,  53,  227, 137, 2,   139, 36,  164, 62,  244, 198, 188, 216, 134, 238,
      144, 236, 133, 164, 224, 232, 244, 78,  56,  135, 135, 136, 239, 74,  38,
      211, 108, 28,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   184, 235, 47,  21,  63,  131, 44,  138, 172, 226,
      76,  79,  63,  61,  169, 239, 249, 107, 91,  189, 160, 68,  163, 31,  175,
      159, 119, 132, 12,  242, 22,  35,  207, 232, 6,   157, 179, 53,  49,  203,
      48,  85,  41,  199, 134, 181, 2,   7,   220, 3,   11,  203, 116, 240, 169,
      227, 161, 216, 224, 155, 227, 118, 196, 37,  74,  66,  189, 21,  211, 53,
      186, 198, 90,  255, 228, 74,  253, 108, 24,  166, 252, 119, 246, 14,  231,
      128, 29,  84,  222, 49,  123, 77,  81,  225, 250, 26,  139, 187, 230, 151,
      101, 85,  101, 102, 22,  192, 212, 74,  249, 57,  220, 201, 42,  68,  5,
      143, 110, 71,  206, 255, 15,  6,   120, 88,  10,  200, 173, 29,  82,  83,
      152, 7,   94,  159, 224, 21,  112, 106, 104, 93,  247, 1,   62,  210, 163,
      221, 32,  169, 184, 82,  239, 15,  152, 210, 8,   209, 69,  38,  199, 41,
      149, 71,  74,  8,   243, 186, 160, 59,  154, 110, 16,  161, 131, 2,   17,
      94,  178, 126, 187, 237, 44,  56,  126, 134, 129, 93,  36,  28,  43,  243,
      112, 3,   150, 21,  45,  201, 107, 4,   39,  129, 145, 20,  54,  101, 22,
      175, 219, 126, 74,  99,  29,  254, 64,  34,  188, 220, 134, 26,  52,  149,
      173, 81,  85,  31,  17,  14,  46,  92,  93,  190, 182, 131, 110, 38,  27,
      150, 218, 44,  36,  215, 116, 49,  131, 57,  5,   171, 232, 160, 153, 194,
      69,  171, 198, 216, 64,  6,   108, 166, 229, 28,  238, 244, 161, 158, 185,
      60,  33,  129, 230, 119, 23,  143, 15,  245, 101, 205, 20,  18,  16,  98,
      242, 54,  162, 135, 211, 40,  209, 35,  82,  81,  252, 151, 133, 131, 184,
      147, 213, 29,  82,  245, 245, 132, 214, 35,  195, 171, 233, 111, 248, 125,
      19,  91,  64,  53,  227, 179, 92,  57,  200, 25,  146, 111, 246, 213, 208,
      41,  243, 240, 174, 63,  194, 246, 243, 42,  83,  86,  172, 152, 17,  197,
      249, 149, 66,  66,  74,  45,  36,  225, 68,  28,  147, 47,  33,  225, 67,
      167, 255, 72,  133, 63,  185, 40,  5,   192, 110, 162, 165, 39,  104, 123,
      39,  184, 230, 227, 246, 44,  6,   165, 94,  173, 162, 133, 41,  20,  238,
      230, 216, 95,  105, 102, 7,   149, 138, 183, 228, 140, 124, 92,  108, 232,
      203, 142, 55,  230, 102, 215, 64,  50,  69,  186, 125, 133, 179, 227, 180,
      8,   181, 46,  174, 81,  174, 59,  208, 166, 255, 49,  207, 166, 216, 179,
      13,  128, 231, 17,  108, 233, 123, 180, 18,  139, 108, 250, 220, 31,  128,
      78,  208, 36,  175, 64,  55,  183, 155, 146, 186, 81,  68,  183, 231, 226,
      83,  248, 197, 135, 115, 56,  211, 163, 17,  244, 21,  251, 104, 175, 234,
      144, 66,  63,  182, 17,  202, 136, 171, 152, 11,  22,  230, 123, 70,  56,
      106, 0,   80,  150, 170, 13,  93,  117, 46,  209, 89,  71,  165, 186, 116,
      58,  45,  120, 100, 34,  10,  2,   181, 23,  60,  231, 128, 117, 94,  6,
      117, 183, 114, 103, 46,  34,  190, 148, 27,  15,  229, 94,  49,  86,  222,
      35,  251, 181, 137, 0,   248, 210, 107, 16,  230, 226, 70,  16,  196, 68,
      82,  52,  36,  72,  226, 87,  40,  41,  35,  75,  12,  143, 40,  127, 112,
      5,   241, 94,  77,  158, 37,  199, 53,  106, 208, 17,  70,  125, 235, 164,
      145, 166, 123, 80,  29,  11,  195, 173, 161, 59,  121, 5,   253, 87,  207,
      127, 35,  199, 56,  186, 86,  173, 73,  41,  101, 170, 105, 45,  187, 203,
      114, 155, 9,   220, 181, 25,  60,  62,  189, 186, 140, 109, 116, 191, 116,
      12,  25,  107, 149, 253, 129, 208, 84,  50,  114, 140, 225, 14,  230, 9,
      101, 114, 6,   42,  225, 168, 144, 130, 118, 127, 192, 132, 89,  97,  195,
      132, 249, 66,  77,  28,  33,  197, 165, 22,  43,  162, 34,  205, 86,  124,
      93,  8,   216, 38,  52,  203, 111, 248, 91,  133, 3,   58,  46,  111, 108,
      84,  62,  19,  146, 228, 29,  206, 165, 50,  95,  237, 61,  50,  100, 160,
      182, 145, 167, 11,  5,   214, 242, 93,  254, 82,  211, 155, 103, 163, 252,
      255, 159, 75,  24,  195, 118, 103, 65,  51,  87,  123, 39,  77,  209, 21,
      252, 88,  170, 253, 87,  14,  181, 242, 212, 248, 90,  139, 4,   29,  188,
      106, 189, 93,  24,  187, 83,  100, 252, 136, 162, 215, 121, 106, 194, 192,
      18,  125, 30,  95,  235, 152, 3,   8,   3,   139, 146, 78,  249, 157, 88,
      253, 38,  38,  125, 67,  96,  100, 145, 252, 10,  196, 97,  232, 115, 218,
      122, 159, 191, 49,  142, 236, 180, 57,  245, 24,  158, 45,  216, 220, 196,
      3,   14,  140, 184, 252, 249, 254, 213, 204, 213, 142, 210, 121, 31,  22,
      230, 171, 68,  58,  0,   215, 176, 228, 198, 26,  103, 1,   198, 85,  203,
      131, 83,  228, 146, 174, 230, 201, 33,  250, 234, 88,  219, 29,  25,  38,
      114, 34,  0,   146, 168, 103, 108, 0,   105, 47,  242, 40,  89,  155, 79,
      209, 88,  253, 136, 149, 249, 78,  251, 52,  164, 133, 223, 62,  71,  15,
      99,  46,  122, 220, 129, 126, 224, 168, 191, 142, 142, 138, 35,  76,  234,
      137,
  };

  // clang-format off
  constexpr static Point kAdviceCommitments[][3] = {
    {
      {"0x02256f3c5db09eaa3ed2f22d335f543f34f9b50f2714271b90bbbbfe8a365ac5",
       "0x1892bac12516e09b25b7acaa45dccc104fb325529df299fcb08b7fc250723b1a"},
      {"0x2e3534433e05f5b1242605ef6432257516e22cb8b5ae1fc2da06b32b9092585d",
       "0x30035ae0c7892f2d414997acee56077efc83c863be8eca1b40b563ef293b5f06"},
      {"0x197f0b5647a76ebf8cfbe549fa3b32da691bbab2217cd7556b336ac0f56cbe54",
       "0x125234fa5387f32f1959d867a9854afe20c9079047f05a483eca4d5535a4bb3c"},
    },
    {
      {"0x02256f3c5db09eaa3ed2f22d335f543f34f9b50f2714271b90bbbbfe8a365ac5",
       "0x1892bac12516e09b25b7acaa45dccc104fb325529df299fcb08b7fc250723b1a"},
      {"0x2e3534433e05f5b1242605ef6432257516e22cb8b5ae1fc2da06b32b9092585d",
       "0x30035ae0c7892f2d414997acee56077efc83c863be8eca1b40b563ef293b5f06"},
      {"0x197f0b5647a76ebf8cfbe549fa3b32da691bbab2217cd7556b336ac0f56cbe54",
       "0x125234fa5387f32f1959d867a9854afe20c9079047f05a483eca4d5535a4bb3c"},
    },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x14f3b93a80e248a02909e66b2eba85707fd93f0704e4c8cfd3f78f01c5c1f16f";

  constexpr static std::string_view kBeta =
      "0x1308cd01f603ad39104efce7c7e4ab02660f5e4a2560a52a544e54255d989c03";

  constexpr static std::string_view kGamma =
      "0x2166363a3350c7ca72d613cc803216b94bb14b897a573748421cff8f75485369";

  // clang-format off
  constexpr static Point kPermutationProductCommitments[][4] = {
      {
          {"0x2cacf6b7c587076fbe217866dae258750d966ed0b9f8283d75385448e74f6c0f",
            "0x17cc0430326288fea27ea90c860e1be5b7b288085a0f60d19a9b72333ad1bf4e"},
          {"0x214a81c1d76766f245cf3d93c7f5275132b1eb3cbe65fb0a00a3edfedbace416",
            "0x09f49fb47cccedc71bbad7498af66b01256a9dc2e8e511b29e72d88f324688af"},
          {"0x047d85cb80c7c7fe22078187268a4cb8407017c02e4902a269c53c3b54b91f31",
            "0x2c95183c5f11dfc9b4109fa3ed22f64432075bcc661f5c5d4ca4c9f5082d9bdc"},
          {"0x1127ac9484656cd7b4937ae0e92877fae8a0a544a8b1e550bae0602b77f9c107",
            "0x00aab73597404e28dc0b410b17176e4e1a18e8dfce1523e2608cd049fa77664b"},
      },
      {
          {"0x0e5032f102fae34df431006a6ca19cacfe320a79695b50f24e7089f36645609b",
            "0x136e8ad75e6513526b2fed4a59224bd424a84c27c3fe990783b228b8aad7d878"},
          {"0x2e4d4248b704a1b81420d0d18d6e5895c8c4e57b7dca18e9548b6a0ec27aaea7",
            "0x07921d99e66d6d6287d42133ace652c1e6ecba049895e867cc7bfd16c066d29d"},
          {"0x24a7e69b88fd83224b484478253624a1a9ed56bfcfac488cc7af9402fb605c7a",
            "0x1ac17d343d1efe795d60c3d23a9a46d6d2508a9c2c92ddb8ef93780f4a0ff45e"},
          {"0x0d650833afedaf6736c25149e25ce8573df1f8d49b101a5daaa6efb20310bf66",
            "0x093dd3d8baea83c598c5dd0597c3486a685c09162ed1cf61f040e36308e0cd0f"},
      },
  };
  // clang-format on

  constexpr static std::string_view kY =
      "0x1460fd3f8816fa92d0992f1d6bf739a43414bb029d725259ae620bf2e8deaaff";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x1797a191ca8ba5610a07855a87e398424b696e7ec2d9e0a1fc59e16dae4167b8",
       "0x041e9a88817cd6f2ccd9792c33eddd65c85d2e8cf48bcafd1010e56df277117f"},
      {"0x0459310bd18163df5f884f46acbce3755c5ba4c05354a0cd6ae029d5837c5d2a",
       "0x184d381109726937f9277181d1fd768e4bdbb16b6a48f29ffb3a46ca714083da"},
  };

  constexpr static std::string_view kX =
      "0x1a31b79708935088f60b0c724b2f04ce2e3f0ff886cfb5717e3d66a2d2da59b0";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x219d5fa867e4fb546592c4592343fadf08385a7322a1dc1d80d7b2bbbb9f90e5",
          "0x3052fc77626e05463d5e056f87cf1d36e6e60e2475d4291ec5a6881c9f83c2b9",
          "0x2ff28cd1a8c381185a474441eee73e0b01c83de07df3aa601154abb7e6e0b621",
      },
      {
          "0x219d5fa867e4fb546592c4592343fadf08385a7322a1dc1d80d7b2bbbb9f90e5",
          "0x3052fc77626e05463d5e056f87cf1d36e6e60e2475d4291ec5a6881c9f83c2b9",
          "0x2ff28cd1a8c381185a474441eee73e0b01c83de07df3aa601154abb7e6e0b621",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x1c6cd3264aef888787384ef4e8e0a485ec90ee86d8bcc6f43ea4248b0289e335",
  };

  constexpr static std::string_view kCommonPermutationEvals[] = {
      "0x2316f20c84779faf1fa344a0bd5b6bf9efa93d3f4f4ce2ac8a2c833f152febb8",
      "0x25c476e39be0d8a1e3a9f074cb0b03dc0702b586c7295530cb3135b39d06e8cf",
      "0x1afae1514d7b31de541d80e70ef677fca6186cfd4ae4ff5ac6ba35d315bd424a",
      "0x1dadc80a5878060fffce476e8f05442ac9dc39f94ad4c0166665556597e6bb8b",
  };

  constexpr static std::string_view kPermutationProductEvals[][4] = {
      {
          "0x29c72645d108d2980fef52b8a920dda3d23e01f75d686a7015e09f5e07985352",
          "0x0640d8c6ab45c299a0e8ab0539833174d7242cda961b266e83b6be5d5c2e0e11",
          "0x2f931c44e1242d4a424295f9c51198ac56532af3f6c23faef0f329d0d5f66f92",
          "0x24d04e801fdcfa6c8b12b47be96c11e7800db3d8a6cf31ffa6d03bae51ae2eb5",
      },
      {
          "0x020a2264782d3a74baa54759d12e755d0daa9650006a38467be6160b98ab88ca",
          "0x2d69aa652949ad56ba38c7237fcf57fd05793ba1adc30b1d507ba691a4eb7d46",
          "0x0ba791b6a064323ded5f32a5ce1de492133e546c6f2e3a03855bf86fcb3426d8",
          "0x18f539b4ec8e31bf9f7ada73e861c40afc916460437d2626fd589df94e928b03",
      },
  };

  constexpr static std::string_view kPermutationProductNextEvals[][4] = {
      {
          "0x0370f32b1c245d81867e382cedbb7eb25e110283a1106e9a3ba0baf3084a4795",
          "0x23d128d387a236f262101214cd65f50f8f1777e681213cb99ea1f4ee1ce5a66c",
          "0x142985a2ad5ea5062cf6e3e6b8277b6827a5a26ec00528b93f8548ffa743e121",
          "0x11b63f4290eaaf68fb15f411a3d3387387c5f853e2e7b74451ba929bb73740af",
      },
      {
          "0x106bd2f80089b5fb23de56315ee50f1b94be222e6772b775065e7580e73c17b5",
          "0x09e60ee18c723254d081fd956b190c74bf746d8cbabd3e3c19b5dc099b72cbbb",
          "0x0e57fdaa58fc15d14d277b5733416776c3184b9ffffca3679bd352fe5df2d605",
          "0x01671ac6e4b0d7003a44abe6161f79d28ed5ccd5fef9fcb88c0e03c4dcd82d9e",
      },
  };

  constexpr static std::string_view kPermutationProductLastEvals[][4] = {
      {
          "0x1f5551ad95341a86dcbc2240fe1d634a7edbaf16653614918127046bc92d1596",
          "0x19c8395cb3e335405b137df86fe9abc323d684f5f5521dd593b8838597fc5152",
          "0x08b4e3b3857dba453240d766e6378ecbe86c5c7c8ce4b78a950766695fd8e6ee",
          "",
      },
      {
          "0x11d06a35c7259e4d5ef105707f288f0c4b23292857e24824345244c41046e2e6",
          "0x085d7c56cd22a22b16a5c5211c4d42f984c3615984c07f768290a8e12a067265",
          "0x080398eb5f1e7d12c0c26a79d7a288fc6453bb185dbd6abc1d048b5af8d4f2b5",
          "",
      },
  };

  static void TestConfig(Fibonacci1Config<F>& config) {
    std::array<AdviceColumnKey, 3> expected_advice = {
        AdviceColumnKey(0),
        AdviceColumnKey(1),
        AdviceColumnKey(2),
    };
    EXPECT_EQ(config.advice, expected_advice);
    EXPECT_EQ(config.instance, InstanceColumnKey(0));
    EXPECT_EQ(config.selector, Selector::Simple(0));
  }

  static std::vector<Evals> GetInstanceColumns() {
    F a(1);
    F b(1);
    F out(55);
    std::vector<F> instance_column = {std::move(a), std::move(b),
                                      std::move(out)};
    return std::vector<Evals>{Evals(std::move(instance_column))};
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI1_CIRCUIT_TEST_DATA_H_

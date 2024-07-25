#ifndef TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/examples/circuit_test_data.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PCS, typename LS>
class Fibonacci2TestData : public CircuitTestData<Circuit, PCS, LS> {
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
        "num_fixed_columns: 0, "
        "num_advice_columns: 1, "
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
                "column_index: 0, "
                "rotation: Rotation(1) "
              "}"
            "), "
            "Negated(Advice { "
              "query_index: 2, "
              "column_index: 0, "
              "rotation: Rotation(2) "
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
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(1)"
          "), "
          "("
            "Column { "
              "index: 0, "
              "column_type: Advice "
            "}, "
            "Rotation(2)"
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
      InstanceColumnKey(0),
  };

  // clang-format off
    constexpr static Label kCycleStoreMapping[][kN] = {
        {{1, 0}, {1, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
         {0, 8}, {1, 2}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
        {{0, 0}, {0, 1},  {0, 9},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
         {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
    };

    constexpr static Label kCycleStoreAux[][kN] = {
        {{0, 0}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
         {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
        {{0, 0}, {0, 1},  {0, 9},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
         {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
    };
  // clang-format on

  constexpr static size_t kCycleStoreSizes[][kN] = {
      {2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  };

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      { true,  true,  true,  true,  true,  true,  true,  true,
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
          "num_advice_columns: 1, "
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
                  "column_index: 0, "
                  "rotation: Rotation(1) "
                "}"
              "), "
              "Negated(Advice { "
                "query_index: 2, "
                "column_index: 0, "
                "rotation: Rotation(2) "
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
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(1)"
            "), "
            "("
              "Column { "
                "index: 0, "
                "column_type: Advice "
              "}, "
              "Rotation(2)"
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
          "(0x179d4bdbaa0d2b6a977d418a8a90d5fe4d9162cf6569dcab6f2a7ab150cf34f7, "
            "0x00638bb01b57c58d95e93d44a6a6bc126133db05161925701f8d0f16bc8f2049), "
          "(0x0356186a0d83b4475fc2e5b01191cff523ee5cc46ef3fa65eff3f611b69aeae6, "
            "0x13b0cc2fd4f1145e719c43d03ff5dfad5c0bb3e0642a5a40ae98f19e6a3b76d7)"
        "] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x0b95ba00d9df3c7f61587edd2ada6ca715d62dc579333da7b80365838146a2a1";

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
          "0x09226b6e22c6f0ca64ec26aad4c86e715b5f898e5e963f25870e56bbe533e9a2",
          "0x133f51f46c2a51201fbb89dc1d6868acfef0f7ce0e572ccb98e04e99eac3ea36",
          "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
          "0x107aab49e65a67f9da9cd2abf78be38bd9dc1d5db39f81de36bcfa5b4b039043",
          "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
          "0x2290ee31c482cf92b79b1944db1c0147635e9004db8c3b9d13644bef31ec3bd3",
          "0x1d59376149b959ccbd157ac850893a6f07c2d99b3852513ab8d01be8e846a566",
          "0x2d8040c3a09c49698c53bfcb514d55a5b39e9b17cb093d128b8783adb8cbd723",
          "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
          "0x1240a374ee6f71e12df10a129946ca9ba824f58a50e3a58b4c68dd5590b74ad8",
          "0x0530d09118705106cbb4a786ead16926d5d174e181a26686af5448492e42a181",
          "0x1fe9a328fad7382fddb3730a89f574d14e57caeac619eeb30d24fb38a4fc6fbe",
          "0x0000000000000000b3c4d79d41a91758cb49c3517c4604a520cff123608fc9cb",
          "0x0dd360411caed09700b52c71a6655715c4d558439e2d34f4307da9a4be13c42e",
          "0x130b17119778465cfb3acaee30f81dee20710ead41671f568b11d9ab07b95a9b",
          "0x02e40daf409556c02bfc85eb303402b774954d30aeb0337eb85a71e6373428de",
      },
      {
          "0x0000000000000000000000000000000000000000000000000000000000000001",
          "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
          "0x0f5c21d0ca65e0db9be1f670eca407d08ec5ec67626a74f892ccebcd0cf9b9f6",
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
  };

  constexpr static std::string_view kPermutationsPolys[][kN] = {
      {
          "0x05f06af4a5e9253f56ae900500cf8fa172aa95f2f378a820db039783f50af1eb",
          "0x0f0fb0d263f5bb14543ae200a13af2e0780d1a4b90f748455c8a62ac18fa3e92",
          "0x04fef3b4de894e2db58283cef83f883f47375989a41b4f09d88f3fdd6834b359",
          "0x1553255b19cd6f6c6f1ca6bd6a63ffbaa9e6d3c4097a6e345e67309580987282",
          "0x0f4081c215e2331724219e823ac5d230430b72fce41edba72edfca2e06ace94e",
          "0x17af7b00c0b54135ad0385ff54daec3e776245036e7e241cb65d6d913f605e7c",
          "0x268fe52e8552462fad2c76e1c436c051a8dadc6faf57a7edb941360f83e46273",
          "0x2622d75b5033efd7da6fc6323de569b738fc901100e416b703b59f3c0a9c8054",
          "0x258ba71da37b24fe773531d00b1aab7e3beec67e42dc6242f143ea34f99b8b4a",
          "0x1c6c613fe56e8f2979a8dfd46aaf483f368c4225a55dc21e6fbd1f0cd5ac3ea4",
          "0x267d1e5d6adafc1018613e0613aab2e0676202e79239bb59f3b841db8671c9dc",
          "0x1628ecb72f96dad15ec71b17a1863b6504b288ad2cda9c2f6de051236e0e0ab3",
          "0x1c3b905033821726a9c22352d12468ef6b8de97452362ebc9d67b78ae7f993e7",
          "0x13cc971188af090820e03bd5b70f4ee13737176dc7d6e64715ea1427af461eb9",
          "0x04ec2ce3c412040e20b74af347b37ace05be800186fd627613064ba96ac21ac2",
          "0x05593ab6f9305a65f373fba2ce04d168759ccc603570f3acc891e27ce409fce1",
      },
      {
          "0x2a73e37e3b487aea61a1b5b180b1c8bbb58952558640c87068de5e0ffaf50e16",
          "0x0d4a403ce45e823da64a1fc2ea0fc49c47fac7f9b1263e28bab23e0cc4c6eb46",
          "0x104b847e4252b02b6ab6bd3fae4e4abc9da4214de88940106d3b6c414ee5fc2d",
          "0x216ab6dee7297be12501e1057228d4aacb4e78d7420e9f7deccb6dcd1a5fcc46",
          "0x06bb7a5b66c3983d3f1ad9724b177b6499dbbb69197ad1f1164b8c375e3ac163",
          "0x0d91c08cd724088d203e3abe0b7ea14663b74b4c4d3f98a9061431536ace10bc",
          "0x02179a5d2e64b0905bba4db5dc0bbff870c2f4fbed3fc2c0930e5f4a87c5649b",
          "0x25e4597d0c05edad7d34f0b090b4dc59ef8978c0d2c9d724737da712d69a5597",
          "0x16952abd7181b114fced42dfe132aa9cd3263ab7f77c44604c630a92e1ecc29b",
          "0x2fa3e6e10afbdbb10f41baed2131914a9b349adc88e50966ffeb6909ce397ed9",
          "0x1db0d924eac719816e0b332c53c4ea538baec560927b75fd0e0746c2dd1be35b",
          "0x2aab9ef338aff970a688c3fc3c7291a06c27b673776c63b7fe0f728064fdcbb2",
          "0x22d48d0fde7027eea7cfc12c4134fc7821119fc8e303cebbab6bde03a7967386",
          "0x293b996a4e467679d49ba46cac3962795815c3ab51870c6a3d55512fca187461",
          "0x0a20cf33b54dd1c458ad0f578e909dcf8153244c983dc41e24c5cafb3eebc176",
          "0x2902b4caa7502955b50a4334306200ed44d1287c67b043b45c8241117cbb180c",
      },
  };

  constexpr static uint8_t kProof[] = {
      102, 195, 181, 202, 144, 137, 33,  216, 68,  179, 205, 95,  227, 168, 162,
      99,  242, 209, 126, 138, 153, 132, 61,  39,  144, 81,  117, 233, 132, 215,
      141, 169, 102, 195, 181, 202, 144, 137, 33,  216, 68,  179, 205, 95,  227,
      168, 162, 99,  242, 209, 126, 138, 153, 132, 61,  39,  144, 81,  117, 233,
      132, 215, 141, 169, 199, 251, 235, 84,  66,  51,  87,  191, 141, 175, 210,
      178, 118, 97,  195, 187, 158, 174, 31,  182, 202, 29,  239, 147, 3,   94,
      157, 61,  185, 67,  181, 31,  32,  65,  40,  125, 62,  194, 31,  196, 239,
      246, 52,  186, 224, 133, 68,  251, 49,  255, 249, 85,  154, 192, 185, 29,
      234, 11,  121, 2,   125, 9,   121, 132, 153, 8,   221, 196, 110, 88,  19,
      58,  221, 161, 188, 20,  134, 199, 164, 26,  198, 159, 55,  86,  246, 253,
      56,  142, 85,  144, 201, 99,  154, 148, 135, 149, 0,   28,  159, 139, 195,
      134, 246, 131, 93,  127, 134, 128, 147, 93,  93,  165, 53,  239, 248, 136,
      73,  190, 181, 44,  35,  167, 64,  108, 46,  111, 8,   168, 1,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   129,
      255, 212, 247, 163, 73,  110, 208, 75,  138, 9,   53,  32,  252, 54,  213,
      190, 31,  116, 90,  71,  189, 44,  117, 159, 222, 6,   61,  221, 40,  164,
      42,  105, 55,  19,  127, 116, 173, 185, 23,  152, 19,  176, 25,  48,  6,
      251, 177, 27,  254, 122, 125, 135, 195, 197, 116, 36,  177, 41,  127, 216,
      103, 154, 171, 125, 197, 228, 28,  10,  184, 186, 161, 247, 161, 91,  7,
      58,  152, 206, 117, 177, 251, 160, 18,  68,  222, 223, 182, 138, 237, 131,
      160, 140, 38,  121, 16,  221, 65,  254, 132, 2,   22,  88,  105, 193, 100,
      170, 102, 65,  255, 55,  80,  187, 150, 167, 129, 125, 28,  24,  50,  159,
      155, 67,  102, 130, 244, 39,  11,  131, 229, 162, 105, 129, 220, 145, 170,
      51,  239, 235, 30,  41,  250, 234, 161, 154, 66,  195, 15,  157, 168, 183,
      10,  139, 170, 60,  12,  2,   107, 81,  33,  125, 197, 228, 28,  10,  184,
      186, 161, 247, 161, 91,  7,   58,  152, 206, 117, 177, 251, 160, 18,  68,
      222, 223, 182, 138, 237, 131, 160, 140, 38,  121, 16,  221, 65,  254, 132,
      2,   22,  88,  105, 193, 100, 170, 102, 65,  255, 55,  80,  187, 150, 167,
      129, 125, 28,  24,  50,  159, 155, 67,  102, 130, 244, 39,  11,  131, 229,
      162, 105, 129, 220, 145, 170, 51,  239, 235, 30,  41,  250, 234, 161, 154,
      66,  195, 15,  157, 168, 183, 10,  139, 170, 60,  12,  2,   107, 81,  33,
      209, 91,  151, 138, 191, 214, 132, 54,  6,   182, 84,  19,  162, 111, 181,
      9,   100, 202, 201, 107, 82,  201, 176, 241, 0,   249, 229, 252, 104, 90,
      39,  33,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   169, 196, 106, 139, 225, 36,  175, 205, 231, 28,  36,
      157, 101, 47,  135, 169, 64,  192, 176, 143, 23,  96,  205, 69,  58,  31,
      142, 211, 101, 54,  252, 6,   85,  239, 31,  161, 14,  201, 1,   22,  226,
      76,  25,  30,  245, 208, 212, 149, 1,   199, 200, 140, 28,  239, 215, 166,
      21,  127, 208, 102, 154, 122, 224, 17,  189, 69,  241, 209, 12,  32,  32,
      132, 189, 221, 119, 132, 15,  210, 225, 252, 116, 55,  61,  161, 78,  255,
      122, 44,  73,  3,   243, 25,  32,  141, 104, 18,  217, 251, 216, 178, 71,
      223, 193, 134, 162, 125, 113, 113, 242, 198, 195, 77,  120, 241, 64,  83,
      57,  127, 80,  215, 63,  21,  119, 117, 185, 101, 227, 37,  74,  77,  210,
      62,  104, 167, 61,  226, 38,  236, 93,  163, 69,  149, 218, 67,  209, 195,
      236, 190, 24,  8,   141, 197, 139, 200, 92,  50,  40,  229, 113, 22,  188,
      196, 133, 103, 238, 206, 224, 83,  33,  171, 99,  81,  83,  215, 94,  198,
      124, 33,  167, 80,  192, 175, 203, 75,  124, 159, 177, 241, 59,  86,  84,
      32,  208, 146, 205, 212, 251, 240, 94,  245, 90,  2,   108, 239, 90,  158,
      17,  118, 197, 8,   137, 152, 91,  246, 247, 250, 40,  46,  129, 227, 151,
      234, 194, 8,   122, 104, 156, 104, 111, 250, 255, 101, 46,  65,  116, 230,
      166, 230, 150, 61,  120, 210, 169, 90,  250, 48,  123, 21,  214, 83,  117,
      174, 225, 95,  67,  11,  244, 11,  243, 189, 144, 54,  235, 74,  244, 40,
      244, 46,  111, 97,  184, 212, 124, 171, 167, 82,  69,  54,  170, 21,  100,
      102, 170, 177, 53,  242, 89,  36,  139, 237, 59,  242, 31,  230, 128, 78,
      44,  213, 243, 181, 35,  117, 59,  46,  2,   195, 64,  158, 211, 227, 53,
      162, 197, 101, 21,  250, 12,  5,   187, 7,   184, 27,  125, 191, 169, 54,
      84,  8,   142, 209, 196, 44,  54,  192, 208, 179, 162, 134, 165, 230, 88,
      193, 194, 145, 176, 102, 153, 225, 166, 145, 16,  27,  65,  3,   44,  203,
      105, 250, 3,   114, 104, 176, 205, 34,  70,  193, 253, 86,  43,  142, 180,
      227, 107, 154, 252, 71,  179, 117, 197, 10,  147, 193, 70,  34,  201, 12,
      167, 189, 208, 222, 198, 100, 189, 112, 208, 234, 3,   223, 217, 128, 255,
      29,  204, 166, 162, 170, 97,  199, 183, 25,  106, 248, 168, 202, 208, 129,
      178, 177, 15,  47,  94,  232, 226, 122, 179, 4,   158, 29,  145, 175, 223,
      16,  11,  224, 233, 205, 29,  0,   252, 211, 68,  190, 160, 156, 15,  129,
      112, 141};

  // clang-format off
  constexpr static Point kAdviceCommitments[][1] = {
    {
      {"0x298dd784e9755190273d84998a7ed1f263a2a8e35fcdb344d8218990cab5c366",
       "0x1d42911d6e4d1799ebf92175f0a9210e911ff8dce85cce34235704fb0fd3c6e3"},
    },
    {
      {"0x298dd784e9755190273d84998a7ed1f263a2a8e35fcdb344d8218990cab5c366",
       "0x1d42911d6e4d1799ebf92175f0a9210e911ff8dce85cce34235704fb0fd3c6e3"},
    },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x1b28757736609214ea9e32e10c911f1b96ae85f38925661ddbd996be266fed09";

  constexpr static std::string_view kBeta =
      "0x0602223cf23aa71e528439ab0e0b6d9a2d931e3a22eba1c870bbeddbadb14983";

  constexpr static std::string_view kGamma =
      "0x00ec0e0eba8b1497529775edac3483f2db08278734c770f45b196fa7133faee3";

  // clang-format off
  constexpr static Point kPermutationProductCommitments[][2] = {
        {
            {"0x1fb543b93d9d5e0393ef1dcab61fae9ebbc36176b2d2af8dbf57334254ebfbc7",
              "0x0e0ee2f2e8a5b8d690bc3fc179c1baa941b1990539d8a4b0d0029cb9a85cf8ea"},
            {"0x0479097d02790bea1db9c09a55f9ff31fb4485e0ba34f6efc41fc23e7d284120",
              "0x2a351db0e33baf97b4bf16b9937a45bbfeabb04d74d5e06d096a9eca57406d55"},
        },
        {
            {"0x1587949a63c990558e38fdf656379fc61aa4c78614bca1dd3a13586ec4dd0899",
              "0x2c5c9cf5d5c2044a43db333083cb3d46a02ac2d2009697ee99fde79f2498bfa7"},
            {"0x28086f2e6c40a7232cb5be4988f8ef35a55d5d9380867f5d83f686c38b9f1c00",
              "0x180a44b0d51f8a1fbdf278a6d2694e6a08f5fb9890b43fac70433fc6f0b62ee5"},
        },
      };
  // clang-format on

  constexpr static std::string_view kY =
      "0x11abab761694e36f90d075f2e4ca450bf11653b3607a676fb9d89621f78e8568";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x2aa428dd3d06de9f752cbd475a741fbed536fc2035098a4bd06e49a3f7d4ff81",
       "0x09a52eb0da623bb6b44e8d58be168483f7a8435700257d8d501827689aa6ee00"},
      {"0x2b9a67d87f29b12474c5c3877d7afe1bb1fb063019b0139817b9ad747f133769",
       "0x24ecbbdabe2de5d884d1fd99e6f7399f0ffce88a0af718d5ac00305aacfc09ff"},
  };

  constexpr static std::string_view kX =
      "0x1cd48fc4021ff60cc4fdf3d4c96109dba22b63fe21c501310c5b3c14f56ce9aa";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x1079268ca083ed8ab6dfde4412a0fbb175ce983a075ba1f7a1bab80a1ce4c57d",
          "0x0b27f48266439b9f32181c7d81a796bb5037ff4166aa64c16958160284fe41dd",
          "0x21516b020c3caa8b0ab7a89d0fc3429aa1eafa291eebef33aa91dc8169a2e583",
      },
      {
          "0x1079268ca083ed8ab6dfde4412a0fbb175ce983a075ba1f7a1bab80a1ce4c57d",
          "0x0b27f48266439b9f32181c7d81a796bb5037ff4166aa64c16958160284fe41dd",
          "0x21516b020c3caa8b0ab7a89d0fc3429aa1eafa291eebef33aa91dc8169a2e583",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x21275a68fce5f900f1b0c9526bc9ca6409b56fa21354b6063684d6bf8a975bd1",
  };

  constexpr static std::string_view kCommonPermutationEvals[] = {
      "0x06fc3665d38e1f3a45cd60178fb0c040a9872f659d241ce7cdaf24e18b6ac4a9",
      "0x11e07a9a66d07f15a6d7ef1c8cc8c70195d4d0f51e194ce21601c90ea11fef55",
  };

  constexpr static std::string_view kPermutationProductEvals[][2] = {
      {
          "0x12688d2019f303492c7aff4ea13d3774fce1d20f8477ddbd8420200cd1f145bd",
          "0x2054563bf1b19f7c4bcbafc050a7217cc65ed7535163ab2153e0ceee6785c4bc",
      },
      {
          "0x0b435fe1ae7553d6157b30fa5aa9d2783d96e6a6e674412e65fffa6f689c687a",
          "0x1b1091a6e19966b091c2c158e6a586a2b3d0c0362cc4d18e085436a9bf7d1bb8",
      },
  };

  constexpr static std::string_view kPermutationProductNextEvals[][2] = {
      {
          "0x25e365b97577153fd7507f395340f1784dc3c6f271717da286c1df47b2d8fbd9",
          "0x08c2ea97e3812e28faf7f65b988908c576119e5aef6c025af55ef0fbd4cd92d0",
      },
      {
          "0x2459f235b1aa666415aa364552a7ab7cd4b8616f2ef428f44aeb3690bdf30bf4",
          "0x2246c1930ac575b347fc9a6be3b48e2b56fdc14622cdb0687203fa69cb2c0341",
      },
  };

  constexpr static std::string_view kPermutationProductLastEvals[][2] = {
      {
          "0x1671e528325cc88bc58d0818beecc3d143da9545a35dec26e23da7683ed24d4a",
          "",
      },
      {
          "0x07bb050cfa1565c5a235e3d39e40c3022e3b7523b5f3d52c4e80e61ff23bed8b",
          "",
      },
  };

  static void TestConfig(Fibonacci2Config<F>& config) {
    EXPECT_EQ(config.advice, AdviceColumnKey(0));
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

#endif  // TACHYON_ZK_PLONK_EXAMPLES_FIBONACCI_FIBONACCI2_CIRCUIT_TEST_DATA_H_

#ifndef TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_TEST_DATA_H_
#define TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_TEST_DATA_H_

#include <stdint.h>

#include <string_view>
#include <utility>
#include <vector>

#include "tachyon/zk/plonk/examples/circuit_test_data.h"
#include "tachyon/zk/plonk/examples/circuit_test_type_traits.h"
#include "tachyon/zk/plonk/examples/point.h"

namespace tachyon::zk::plonk {

template <typename Circuit, typename PS, typename SFINAE = void>
class ShuffleAPICircuitTestData : public CircuitTestData<Circuit, PS> {};

// PCS = SHPlonk
template <typename Circuit, typename PS>
class ShuffleAPICircuitTestData<Circuit, PS,
                                std::enable_if_t<IsSHPlonk<typename PS::PCS>>>
    : public CircuitTestData<Circuit, PS> {
 public:
  using PCS = typename PS::PCS;
  using F = typename PCS::Field;

  // Set flags of values to be used as true
  constexpr static bool kAssemblyFixedColumnsFlag = true;
  constexpr static bool kAssemblyPermutationColumnsFlag = false;
  constexpr static bool kLFirstFlag = true;
  constexpr static bool kLLastFlag = true;
  constexpr static bool kLActiveRowFlag = true;
  constexpr static bool kFixedColumnsFlag = true;
  constexpr static bool kFixedPolysFlag = true;
  constexpr static bool kAdviceCommitmentsFlag = true;
  constexpr static bool kShuffleProductCommitmentsFlag = true;
  constexpr static bool kVanishingHPolyCommitmentsFlag = true;
  constexpr static bool kAdviceEvalsFlag = true;
  constexpr static bool kFixedEvalsFlag = true;
  constexpr static bool kShuffleProductEvalsFlag = true;
  constexpr static bool kShuffleProductNextEvalsFlag = true;

  constexpr static size_t kN = 16;

  // clang-format off
  constexpr static std::string_view kPinnedConstraintSystem =
      "PinnedConstraintSystem { "
        "num_fixed_columns: 1, "
        "num_advice_columns: 3, "
        "num_instance_columns: 0, "
        "num_selectors: 2, "
        "gates: [], "
        "advice_queries: [("
          "Column { "
              "index: 0, "
              "column_type: Advice "
          "}, "
          "Rotation(0)"
        "), ("
          "Column { "
              "index: 1, "
              "column_type: Advice "
          "}, "
          "Rotation(0)"
        "), ("
          "Column { "
              "index: 2, "
              "column_type: Advice "
          "}, "
          "Rotation(0)"
        ")], "
        "instance_queries: [], "
        "fixed_queries: [("
          "Column { "
              "index: 0, "
              "column_type: Fixed "
          "}, "
          "Rotation(0)"
        ")], "
        "permutation: Argument { columns: [] }, "
        "lookups_map: {}, "
        "constants: [], "
        "minimum_degree: None "
      "}";
  // clang-format on

  constexpr static std::string_view kAssemblyFixedColumns[][kN] = {
      {
          "0x000000000000000000000000000000000000000000000000000000000000000a",
          "0x0000000000000000000000000000000000000000000000000000000000000014",
          "0x0000000000000000000000000000000000000000000000000000000000000028",
          "0x000000000000000000000000000000000000000000000000000000000000000a",
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

  // clang-format off
  constexpr static bool kSelectors[][kN] = {
      {true,   true,  true,  true, false, false, false, false,
       false, false, false, false, false, false, false, false},
      {true,   true,  true,  true, false, false, false, false,
       false, false, false, false, false, false, false, false},
  };
  // clang-format on

  // clang-format off
  constexpr static std::string_view kPinnedVerifyingKey =
      "PinnedVerificationKey { "
        "base_modulus: \"0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47\", "
        "scalar_modulus: \"0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001\", "
        "domain: PinnedEvaluationDomain { "
            "k: 4, "
            "extended_k: 6, "
            "omega: 0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b "
        "}, "
        "cs: PinnedConstraintSystem { "
            "num_fixed_columns: 3, "
            "num_advice_columns: 3, "
            "num_instance_columns: 0, "
            "num_selectors: 2, "
            "gates: [], "
            "advice_queries: [("
              "Column { "
                  "index: 0, "
                  "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), ("
              "Column { "
                  "index: 1, "
                  "column_type: Advice "
              "}, "
              "Rotation(0)"
            "), ("
               "Column { "
                  "index: 2, "
                  "column_type: Advice "
                "}, "
                "Rotation(0)"
            ")], "
            "instance_queries: [], "
            "fixed_queries: [("
              "Column { "
                  "index: 0, "
                  "column_type: Fixed "
              "}, "
              "Rotation(0)"
            "), ("
              "Column { "
                "index: 1, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            "), ("
              "Column { "
                "index: 2, "
                "column_type: Fixed "
              "}, "
              "Rotation(0)"
            ")], "
            "permutation: Argument { columns: [] }, "
            "lookups_map: {}, "
            "constants: [], "
            "minimum_degree: None "
        "}, "
        "fixed_commitments: ["
            "(0x2d4e15ed2b857e7804667e13cc75167d4d96b17e0b4f5b550a5b2973ce42b38b, "
              "0x202b370f2deb277586ef975cf01b528ab0aa32750ff474a30736990862079124), "
            "(0x281d9f77f3053cb0fab1a344caab67658751a45d7878c1a82e1641aad602dc4f, "
              "0x2bc66678fb3b22f2fbf7352fc01959f0db70006764124d957655af470c9048d2), "
            "(0x281d9f77f3053cb0fab1a344caab67658751a45d7878c1a82e1641aad602dc4f, "
              "0x2bc66678fb3b22f2fbf7352fc01959f0db70006764124d957655af470c9048d2)"
        "], "
        "permutation: VerifyingKey { commitments: [] } "
      "}";
  // clang-format on

  constexpr static std::string_view kTranscriptRepr =
      "0x251c2c983235c1ee7172d0394aba3b41563ff49d03e46c40d9802c72c95ceecb";

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
          "0x000000000000000000000000000000000000000000000000000000000000000a",
          "0x0000000000000000000000000000000000000000000000000000000000000014",
          "0x0000000000000000000000000000000000000000000000000000000000000028",
          "0x000000000000000000000000000000000000000000000000000000000000000a",
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
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
      {
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
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
      },
  };

  constexpr static std::string_view kFixedPolys[][kN] = {
      {
          "0x0000000000000000000000000000000000000000000000000000000000000005",
          "0x29b74984a8e202a76661c45c0e1c23834ec7253d1314c598e6f77c7ad51cf036",
          "0x02da37f73c03ba855e9e5e8bf35f1551564eaf8cfc98ba948b81dd51fff8ceae",
          "0x228042fcf4bc2d1ad9b24f632d19dd1548bf7f52ebd79064e83380477cbeaf72",
          "0x0000000000000000705b06c24909ae977f0e1a12edabc2e73481f6b61c59de1d",
          "0x251f9565fdb87df90efa68e06fd185b8c4d1ca853f8cb18f855fbac180d014f2",
          "0x064adf83a4eb29366c2663c6bf99c5b0194718e855ce0bfd4f142a28f13faa07",
          "0x08058968c83a758fcb08defe026a6ac37dcf2b050519bcdf6fefd2f549af5a88",
          "0x244b3ad628e5381f4a3c3448e1210245de26ee365b4b146cf2e9782ef4000002",
          "0x29671636942af51f13b020dd465c1c0c09612a1315807af37f6946775581d4d2",
          "0x217102deece17d9d6e760bcf360f61b080490f40cc1071120b77508dd6d6224b",
          "0x1bbf0aaeae5aa0266b112a884a1e32f2c27eb74af79f84a9775b61555e8e7815",
          "0x244b3ad628e5381ed9e12d86981753ae5f18d4236d9f5185be678178d7a621e0",
          "0x00bccefdf8369aa8c3bc602e7f7997a0c787aadfc82da74f3df4683e2c91260b",
          "0x1e005b5283fa0ee55b3d9a6fd939c7d9cc6f04b6981ef135ffc598551ff16504",
          "0x02519e97e6792d6d84ac0fa8489f89c033407ac9ce0556ec16543bcbc3037df5",
      },
      {
          "0x244b3ad628e5381f4a3c3448e1210245de26ee365b4b146cf2e9782ef4000001",
          "0x1a6e4f898fe496f01ea6eaaff139c3aa49b7af6655c100c5392f9a1d57d0147b",
          "0x19b5e5b39b97598b23d3677c6697d6157032b89240d1b0f0c7a45bfb7168bc5f",
          "0x2b59c6d6b45408d25b4d11e087615929d58e833eb19438fedddd2ee1ed5223ce",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x05e4b072c5b5c514ab7373d186f5bffccef88a06715bb3b86770a20e870495b3",
          "0x2bdb831eb009f59ab278e6ad2ef3344d45dcf7433eee7a929cff19ee7f56c326",
          "0x26ec8a1cb3b2049f951a406829dd837d8e9bb25d65b9343f41f3521abc3b3884",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x00253a9213a341f05ce8918915a5ec420e7066938f489b6e3a239be2012716d9",
          "0x2ee08ff8b63316b3871d9c0903df516165845c44a54d387dc2489286e2a93cdc",
          "0x29fbdc8478d0d99780b5fd0fb89b2e9e0b8681dc53a498d02ed8c7ea1c763064",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x03d30047bfa79a2a2339443e534b925cb7064e3604e5c4811825a02114043efa",
          "0x10a1def0e97412995d72d5834ab9573c12fab2ad29b0918e53c15ae6009743a2",
          "0x08d1aa440a71916949bf705d7a49a7babede074ee3cbef812c041a700dfc734d",
      },
      {
          "0x244b3ad628e5381f4a3c3448e1210245de26ee365b4b146cf2e9782ef4000001",
          "0x1a6e4f898fe496f01ea6eaaff139c3aa49b7af6655c100c5392f9a1d57d0147b",
          "0x19b5e5b39b97598b23d3677c6697d6157032b89240d1b0f0c7a45bfb7168bc5f",
          "0x2b59c6d6b45408d25b4d11e087615929d58e833eb19438fedddd2ee1ed5223ce",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x05e4b072c5b5c514ab7373d186f5bffccef88a06715bb3b86770a20e870495b3",
          "0x2bdb831eb009f59ab278e6ad2ef3344d45dcf7433eee7a929cff19ee7f56c326",
          "0x26ec8a1cb3b2049f951a406829dd837d8e9bb25d65b9343f41f3521abc3b3884",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x00253a9213a341f05ce8918915a5ec420e7066938f489b6e3a239be2012716d9",
          "0x2ee08ff8b63316b3871d9c0903df516165845c44a54d387dc2489286e2a93cdc",
          "0x29fbdc8478d0d99780b5fd0fb89b2e9e0b8681dc53a498d02ed8c7ea1c763064",
          "0x0000000000000000000000000000000000000000000000000000000000000000",
          "0x03d30047bfa79a2a2339443e534b925cb7064e3604e5c4811825a02114043efa",
          "0x10a1def0e97412995d72d5834ab9573c12fab2ad29b0918e53c15ae6009743a2",
          "0x08d1aa440a71916949bf705d7a49a7babede074ee3cbef812c041a700dfc734d",
      },
  };

  constexpr static uint8_t kProof[] = {
      123, 149, 1,   34,  4,   178, 140, 187, 165, 136, 202, 244, 62,  2,   151,
      119, 141, 172, 38,  112, 93,  191, 240, 213, 173, 141, 104, 35,  108, 108,
      178, 19,  99,  199, 7,   207, 154, 138, 8,   187, 113, 59,  4,   131, 197,
      184, 34,  241, 221, 127, 180, 101, 40,  230, 153, 79,  105, 139, 51,  173,
      113, 30,  0,   36,  122, 60,  229, 71,  171, 195, 59,  222, 207, 84,  230,
      206, 151, 175, 95,  127, 4,   120, 157, 12,  132, 58,  230, 10,  175, 158,
      93,  254, 102, 38,  70,  10,  123, 149, 1,   34,  4,   178, 140, 187, 165,
      136, 202, 244, 62,  2,   151, 119, 141, 172, 38,  112, 93,  191, 240, 213,
      173, 141, 104, 35,  108, 108, 178, 19,  99,  199, 7,   207, 154, 138, 8,
      187, 113, 59,  4,   131, 197, 184, 34,  241, 221, 127, 180, 101, 40,  230,
      153, 79,  105, 139, 51,  173, 113, 30,  0,   36,  122, 60,  229, 71,  171,
      195, 59,  222, 207, 84,  230, 206, 151, 175, 95,  127, 4,   120, 157, 12,
      132, 58,  230, 10,  175, 158, 93,  254, 102, 38,  70,  10,  40,  248, 165,
      230, 155, 94,  21,  42,  243, 43,  4,   150, 132, 65,  71,  208, 235, 189,
      87,  173, 66,  88,  193, 46,  179, 106, 15,  24,  187, 2,   203, 103, 179,
      78,  185, 253, 180, 227, 106, 238, 252, 147, 223, 168, 28,  84,  103, 253,
      68,  13,  163, 107, 92,  33,  77,  82,  222, 210, 85,  156, 0,   144, 90,
      27,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   105, 112, 247, 230, 99,  68,  226, 246, 1,   139, 135, 253,
      42,  40,  223, 143, 240, 44,  114, 89,  116, 112, 83,  29,  171, 147, 13,
      113, 127, 69,  120, 18,  171, 203, 78,  146, 126, 138, 144, 126, 38,  59,
      201, 84,  49,  63,  95,  119, 88,  50,  69,  9,   111, 240, 86,  108, 227,
      3,   62,  20,  16,  170, 144, 80,  90,  201, 15,  47,  196, 4,   58,  104,
      106, 35,  15,  171, 132, 32,  249, 8,   210, 85,  33,  159, 147, 145, 181,
      59,  6,   78,  166, 209, 122, 13,  142, 103, 169, 97,  43,  75,  89,  167,
      101, 233, 61,  72,  185, 239, 7,   16,  199, 229, 75,  28,  130, 185, 165,
      122, 205, 187, 114, 165, 17,  11,  62,  190, 91,  44,  209, 149, 118, 224,
      113, 214, 75,  198, 60,  59,  144, 31,  13,  128, 129, 79,  211, 15,  183,
      118, 231, 213, 108, 68,  15,  66,  202, 65,  78,  236, 109, 41,  132, 133,
      165, 158, 191, 193, 66,  11,  235, 199, 153, 168, 232, 177, 144, 89,  6,
      128, 219, 249, 179, 170, 105, 162, 192, 156, 250, 132, 238, 102, 135, 44,
      169, 97,  43,  75,  89,  167, 101, 233, 61,  72,  185, 239, 7,   16,  199,
      229, 75,  28,  130, 185, 165, 122, 205, 187, 114, 165, 17,  11,  62,  190,
      91,  44,  209, 149, 118, 224, 113, 214, 75,  198, 60,  59,  144, 31,  13,
      128, 129, 79,  211, 15,  183, 118, 231, 213, 108, 68,  15,  66,  202, 65,
      78,  236, 109, 41,  132, 133, 165, 158, 191, 193, 66,  11,  235, 199, 153,
      168, 232, 177, 144, 89,  6,   128, 219, 249, 179, 170, 105, 162, 192, 156,
      250, 132, 238, 102, 135, 44,  254, 142, 96,  254, 124, 46,  191, 114, 233,
      1,   222, 185, 106, 101, 1,   177, 181, 78,  108, 159, 195, 148, 109, 160,
      224, 212, 141, 238, 15,  216, 187, 32,  94,  164, 148, 212, 21,  95,  124,
      142, 88,  181, 210, 147, 54,  249, 68,  116, 40,  166, 135, 251, 94,  53,
      123, 127, 122, 169, 236, 211, 188, 16,  206, 9,   94,  164, 148, 212, 21,
      95,  124, 142, 88,  181, 210, 147, 54,  249, 68,  116, 40,  166, 135, 251,
      94,  53,  123, 127, 122, 169, 236, 211, 188, 16,  206, 9,   1,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   21,
      33,  39,  152, 110, 1,   106, 66,  193, 30,  33,  236, 208, 160, 243, 0,
      227, 17,  226, 232, 246, 24,  184, 154, 9,   99,  110, 202, 176, 8,   127,
      38,  171, 201, 131, 218, 180, 3,   225, 154, 17,  127, 114, 216, 120, 129,
      255, 250, 19,  172, 225, 166, 172, 94,  60,  225, 187, 236, 150, 108, 29,
      178, 211, 36,  197, 110, 67,  66,  4,   243, 99,  92,  118, 119, 166, 253,
      145, 19,  233, 20,  198, 233, 26,  113, 47,  123, 174, 100, 77,  147, 34,
      39,  205, 78,  189, 27,  190, 142, 41,  79,  1,   253, 107, 168, 226, 56,
      123, 143, 162, 88,  115, 99,  145, 164, 236, 237, 250, 62,  225, 212, 44,
      123, 233, 130, 218, 205, 135, 32,  93,  168, 98,  215, 47,  38,  123, 245,
      82,  171, 140, 93,  190, 27,  17,  109, 253, 129, 176, 223, 15,  199, 228,
      215, 184, 224, 79,  137, 196, 110, 11,  42,  211, 42,  183, 215, 47,  79,
      178, 182, 206, 51,  36,  81,  146, 77,  234, 35,  225, 213, 241, 16,  165,
      86,  174, 254, 67,  240, 131, 144, 79,  159, 201, 89};

  // clang-format off
  constexpr static Point kAdviceCommitments[][3] = {
      {
          {"0x13b26c6c23688dadd5f0bf5d7026ac8d7797023ef4ca88a5bb8cb2042201957b",
            "0x0ff9241bc89e18ea71a0890137187b79870096231e31ce087de76e6440773b38"},
          {"0x24001e71ad338b694f99e62865b47fddf122b8c583043b71bb088a9acf07c763",
            "0x16cb64579b12bff951032189709221da1908ac980fcdcecc3058ffe0b331b178"},
          {"0x0a462666fe5d9eaf0ae63a840c9d78047f5faf97cee654cfde3bc3ab47e53c7a",
            "0x1a4978eec4c46d3bf9b9f8012ffe2a8296d1bdf2f702a319fc8439906ca8225e"},
      },
      {
          {"0x13b26c6c23688dadd5f0bf5d7026ac8d7797023ef4ca88a5bb8cb2042201957b",
            "0x0ff9241bc89e18ea71a0890137187b79870096231e31ce087de76e6440773b38"},
          {"0x24001e71ad338b694f99e62865b47fddf122b8c583043b71bb088a9acf07c763",
            "0x16cb64579b12bff951032189709221da1908ac980fcdcecc3058ffe0b331b178"},
          {"0x0a462666fe5d9eaf0ae63a840c9d78047f5faf97cee654cfde3bc3ab47e53c7a",
            "0x1a4978eec4c46d3bf9b9f8012ffe2a8296d1bdf2f702a319fc8439906ca8225e"},
      },
  };
  // clang-format on

  constexpr static std::string_view kTheta =
      "0x209655a8aa61cc6fa4a5b5c538dee9c2e45d96e393ac98518a826f2b65096b1d";

  constexpr static std::string_view kBeta =
      "0x25cda161adf9c679a75f7b870fb2e6aa7c75edce5aa790a546365bc564a93172";

  constexpr static std::string_view kGamma =
      "0x1704848a68c783bb204963d0665ea280ce1c73fc7e18f7ac37a4576df63affa2";

  // clang-format off
  constexpr static Point kShuffleProductCommitments[][1] = {
      {
          {"0x27cb02bb180f6ab32ec15842ad57bdebd047418496042bf32a155e9be6a5f828",
           "0x0ec8207c93ae2600b61d6357c695af63ca92ed723b53bc721227d32be74b11bb"},
      },
      {
          {"0x1b5a90009c55d2de524d215c6ba30d44fd67541ca8df93fcee6ae3b4fdb94eb3",
           "0x26231bd570fbec74da443a9ce5c93e3e933d008032489dbe112560b7796643f2"},
      },
  };
  // clang-format on

  constexpr static std::string_view kY =
      "0x12d2c7429f3b85194400604f68d427d09df281205ab991831cbc2a290e88b7c4";

  constexpr static Point kVanishingHPolyCommitments[] = {
      {"0x1278457f710d93ab1d53707459722cf08fdf282afd878b01f6e24463e6f77069",
       "0x00db9c6dc837d2477649c858851349adb281949a5f1651b89af437088b5ac17c"},
      {"0x1090aa10143e03e36c56f06f09453258775f3f3154c93b267e908a7e924ecbab",
       "0x07845cc1fdf395d5a63a86fcf6a767829e9ca29ddf0264be6904d64d758a3a35"},
      {"0x278e0d7ad1a64e063bb591939f2155d208f92084ab0f236a683a04c42f0fc95a",
       "0x0e3ea5528ad480e93a1bdacdf87fc195153c45a1c57b69066f4869f5254aa8c9"},
  };

  constexpr static std::string_view kX =
      "0x2ae1c2c898906cdb9db5361d365669b204f5d3f580c0fa3384938fb05825e7e0";

  constexpr static std::string_view kAdviceEvals[][3] = {
      {
          "0x2c5bbe3e0b11a572bbcd7aa5b9821c4be5c71007efb9483de965a7594b2b61a9",
          "0x296dec4e41ca420f446cd5e776b70fd34f81800d1f903b3cc64bd671e07695d1",
          "0x2c8766ee84fa9cc0a269aab3f9db80065990b1e8a899c7eb0b42c1bf9ea58584",
      },
      {
          "0x2c5bbe3e0b11a572bbcd7aa5b9821c4be5c71007efb9483de965a7594b2b61a9",
          "0x296dec4e41ca420f446cd5e776b70fd34f81800d1f903b3cc64bd671e07695d1",
          "0x2c8766ee84fa9cc0a269aab3f9db80065990b1e8a899c7eb0b42c1bf9ea58584",
      },
  };

  constexpr static std::string_view kFixedEvals[] = {
      "0x20bbd80fee8dd4e0a06d94c39f6c4eb5b101656ab9de01e972bf2e7cfe608efe",
      "0x09ce10bcd3eca97a7f7b355efb87a6287444f93693d2b5588e7c5f15d494a45e",
      "0x09ce10bcd3eca97a7f7b355efb87a6287444f93693d2b5588e7c5f15d494a45e",
  };

  constexpr static std::string_view kShuffleProductEvals[][1] = {
      {
          "0x267f08b0ca6e63099ab818f6e8e211e300f3a0d0ec211ec1426a016e98272115",
      },
      {
          "0x1bbd4ecd2722934d64ae7b2f711ae9c614e91391fda677765c63f30442436ec5",
      },
  };

  constexpr static std::string_view kShuffleProductNextEvals[][1] = {
      {
          "0x24d3b21d6c96ecbbe13c5eaca6e1ac13faff8178d8727f119ae103b4da83c9ab",
      },
      {
          "0x2087cdda82e97b2cd4e13efaedeca491637358a28f7b38e2a86bfd014f298ebe",
      },
  };

  constexpr static std::string_view kHEval =
      "0x0abd11871bf0169fd16294d572944009f8e98aab3c3cfd9bd8c7d8d772605d67";

  static void TestConfig(ShuffleAPIConfig<F>& config) {
    EXPECT_EQ(config.input_0(), AdviceColumnKey(0));
    EXPECT_EQ(config.input_1(), FixedColumnKey(0));
    EXPECT_EQ(config.shuffle_0(), AdviceColumnKey(1));
    EXPECT_EQ(config.shuffle_1(), AdviceColumnKey(2));
    EXPECT_EQ(config.s_input(), Selector::Complex(1));
    EXPECT_EQ(config.s_shuffle(), Selector::Complex(0));
  }

  static Circuit GetCircuit() {
    std::vector<Value<F>> input_0 = {
        Value<F>::Known(F(1)),
        Value<F>::Known(F(2)),
        Value<F>::Known(F(4)),
        Value<F>::Known(F(1)),
    };
    std::vector<F> input_1 = {
        F(10),
        F(20),
        F(40),
        F(10),
    };
    std::vector<Value<F>> shuffle_0 = {
        Value<F>::Known(F(4)),
        Value<F>::Known(F(1)),
        Value<F>::Known(F(1)),
        Value<F>::Known(F(2)),
    };
    std::vector<Value<F>> shuffle_1 = {
        Value<F>::Known(F(40)),
        Value<F>::Known(F(10)),
        Value<F>::Known(F(10)),
        Value<F>::Known(F(20)),
    };
    return {std::move(input_0), std::move(input_1), std::move(shuffle_0),
            std::move(shuffle_1)};
  }

  static std::vector<Circuit> Get2Circuits() {
    Circuit circuit = GetCircuit();
    return {circuit, std::move(circuit)};
  }
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_SHUFFLE_API_CIRCUIT_TEST_DATA_H_

#include "tachyon/zk/plonk/examples/shuffle_circuit.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/shuffle_circuit_test_data.h"
#include "tachyon/zk/plonk/halo2/pinned_constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

// clang-format off
std::vector<std::vector<std::vector<std::string_view>>> original_tables = {
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

std::vector<std::vector<std::vector<std::string_view>>> shuffled_tables = {
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
  }
};
// clang-format on

constexpr size_t W = 2;
constexpr size_t H = 8;
constexpr size_t K = 4;

class ShuffleCircuitTest : public CircuitTest {
 public:
  static std::vector<std::vector<F>> CreateTable(
      const std::vector<std::vector<std::string_view>>& table) {
    return base::CreateVector(W, [&table](size_t i) {
      std::vector<std::string_view> table_i = table[i];
      return base::CreateVector(
          H, [&table_i](size_t j) { return F::FromHexString(table_i[j]); });
    });
  }

  template <template <typename> class FloorPlanner>
  static ShuffleCircuit<F, W, H, FloorPlanner> GetCircuitForTest(size_t i = 0) {
    CHECK_LT(i, original_tables.size());
    return {CreateTable(original_tables[i]), CreateTable(shuffled_tables[i])};
  }
};

}  // namespace

TEST_F(ShuffleCircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  ShuffleCircuitConfig<F, W> config =
      ShuffleCircuit<F, W, H, SimpleFloorPlanner>::Configure(constraint_system);
  EXPECT_EQ(config.q_shuffle(), Selector::Simple(0));
  EXPECT_EQ(config.q_first(), Selector::Simple(1));
  EXPECT_EQ(config.q_last(), Selector::Simple(2));
  for (size_t i = 0; i < W; i++) {
    EXPECT_EQ(config.original_column_keys()[i], AdviceColumnKey(i));
    EXPECT_EQ(config.shuffled_column_keys()[i], AdviceColumnKey(W + i));
  }
  EXPECT_EQ(config.theta(), Challenge(0, kFirstPhase));
  EXPECT_EQ(config.gamma(), Challenge(1, kFirstPhase));
  EXPECT_EQ(config.z(), AdviceColumnKey(2 * W, kSecondPhase));

  halo2::PinnedConstraintSystem<F> pinned_constraint_system(constraint_system);
  EXPECT_EQ(shuffle::kPinnedConstraintSystem,
            base::ToRustDebugString(pinned_constraint_system));

  EXPECT_TRUE(constraint_system.selector_map().empty());
  EXPECT_TRUE(constraint_system.general_column_annotations().empty());
}

TEST_F(ShuffleCircuitTest, Synthesize) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));
  const Domain* domain = prover_->domain();

  ConstraintSystem<F> constraint_system;
  ShuffleCircuitConfig<F, W> config =
      ShuffleCircuit<F, W, H, SimpleFloorPlanner>::Configure(constraint_system);
  Assembly<RationalEvals> assembly =
      VerifyingKey<F, Commitment>::CreateAssembly<RationalEvals>(
          domain, constraint_system);

  ShuffleCircuit<F, W, H, SimpleFloorPlanner> circuit =
      GetCircuitForTest<SimpleFloorPlanner>();
  typename ShuffleCircuit<F, W, H, SimpleFloorPlanner>::FloorPlanner
      floor_planner;
  floor_planner.Synthesize(&assembly, circuit, std::move(config),
                           constraint_system.constants());

  EXPECT_TRUE(assembly.fixed_columns().empty());

  EXPECT_TRUE(assembly.permutation().columns().empty());

  const CycleStore& cycle_store = assembly.permutation().cycle_store();
  EXPECT_TRUE(cycle_store.mapping().IsEmpty());
  EXPECT_TRUE(cycle_store.aux().IsEmpty());
  EXPECT_TRUE(cycle_store.sizes().IsEmpty());

  // clang-format off
  std::vector<std::vector<bool>> expected_selectors = {
      {true, true, true, true, true, true, true, true,
        false, false, false, false, false, false, false, false},
      {true, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false},
      {false, false, false, false, false, false, false, false,
        true, false, false, false, false, false, false, false}};
  // clang-format on
  EXPECT_EQ(assembly.selectors(), expected_selectors);
  EXPECT_EQ(assembly.usable_rows(), base::Range<RowIndex>::Until(10));
}

TEST_F(ShuffleCircuitTest, LoadVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  ShuffleCircuit<F, W, H, SimpleFloorPlanner> circuit =
      GetCircuitForTest<SimpleFloorPlanner>();

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  halo2::PinnedVerifyingKey pinned_vkey(prover_.get(), vkey);
  EXPECT_EQ(shuffle::kPinnedVerifyingKey, base::ToRustDebugString(pinned_vkey));

  F expected_transcript_repr = F::FromHexString(
      "0x220eecfcd245ace6db06b0c892cef067bb85cc36dc0401c6e05b618bb939f761");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

TEST_F(ShuffleCircuitTest, LoadProvingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  ShuffleCircuit<F, W, H, SimpleFloorPlanner> circuit =
      GetCircuitForTest<SimpleFloorPlanner>();

  for (size_t i = 0; i < 2; ++i) {
    ProvingKey<Poly, Evals, Commitment> pkey;
    bool load_verifying_key = i == 0;
    SCOPED_TRACE(
        absl::Substitute("load_verifying_key: $0", load_verifying_key));
    if (load_verifying_key) {
      VerifyingKey<F, Commitment> vkey;
      ASSERT_TRUE(vkey.Load(prover_.get(), circuit));
      ASSERT_TRUE(
          pkey.LoadWithVerifyingKey(prover_.get(), circuit, std::move(vkey)));
    } else {
      ASSERT_TRUE(pkey.Load(prover_.get(), circuit));
    }

    Poly expected_l_first;
    {
      std::vector<std::string_view> poly = {
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
      expected_l_first = CreatePoly(poly);
    }
    EXPECT_EQ(pkey.l_first(), expected_l_first);

    Poly expected_l_last;
    {
      std::vector<std::string_view> poly = {
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
      expected_l_last = CreatePoly(poly);
    }
    EXPECT_EQ(pkey.l_last(), expected_l_last);

    Poly expected_l_active_row;
    {
      std::vector<std::string_view> poly = {
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
      expected_l_active_row = CreatePoly(poly);
    }
    EXPECT_EQ(pkey.l_active_row(), expected_l_active_row);

    std::vector<Evals> expected_fixed_columns;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> evals = {{
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
      }};
      // clang-format on
      expected_fixed_columns = CreateColumns(evals);
    }
    EXPECT_EQ(pkey.fixed_columns(), expected_fixed_columns);

    std::vector<Poly> expected_fixed_polys;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> polys = {{
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
      }};
      // clang-format on
      expected_fixed_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.fixed_polys(), expected_fixed_polys);

    EXPECT_TRUE(pkey.permutation_proving_key().permutations().empty());
    EXPECT_TRUE(pkey.permutation_proving_key().polys().empty());
  }
}

TEST_F(ShuffleCircuitTest, CreateProof) {
  size_t n = size_t{1} << K;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  std::vector<ShuffleCircuit<F, W, H, SimpleFloorPlanner>> circuits = {
      GetCircuitForTest<SimpleFloorPlanner>(0),
      GetCircuitForTest<SimpleFloorPlanner>(1)};

  std::vector<Evals> instance_columns;
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuits[0]));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(shuffle::kExpectedProof),
                                      std::end(shuffle::kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(ShuffleCircuitTest, CreateV1Proof) {
  size_t n = size_t{1} << K;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  std::vector<ShuffleCircuit<F, W, H, V1FloorPlanner>> circuits = {
      GetCircuitForTest<V1FloorPlanner>(0),
      GetCircuitForTest<V1FloorPlanner>(1)};

  std::vector<Evals> instance_columns;
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuits[0]));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(shuffle_v1::kExpectedProof),
                                      std::end(shuffle_v1::kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(ShuffleCircuitTest, VerifyProof) {
  size_t n = size_t{1} << K;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  ShuffleCircuit<F, W, H, SimpleFloorPlanner> circuit =
      GetCircuitForTest<SimpleFloorPlanner>();

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(shuffle::kExpectedProof),
                                   std::end(shuffle::kExpectedProof));
  Verifier<PCS> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));

  std::vector<Evals> instance_columns;
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  size_t num_circuits = instance_columns_vec.size();
  Proof<F, Commitment> proof;
  F h_eval;
  ASSERT_TRUE(verifier.VerifyProofForTesting(vkey, instance_columns_vec, &proof,
                                             &h_eval));

  std::vector<std::vector<Commitment>> expected_advice_commitments_vec;
  {
    std::vector<Point> points = {
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
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));

    points = {
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
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  std::vector<F> expected_challenges;
  {
    std::vector<std::string_view> challenges = {
        "0x2f2cdac7ef66f10d3c910f13e5a4344318eeeceae9edcdd75b4c1976de43475e",
        "0x154c67d8fa9170f88d424e7634e151902bcf91212c5f7192b58094f50f4da90c"};
    expected_challenges = CreateEvals(challenges);
  }
  EXPECT_EQ(proof.challenges, expected_challenges);

  F expected_theta = F::FromHexString(
      "0x1b05d778b77bda225d239dffc14f3976e8fe1760fd21713b610c3f213589400f");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_permuted_commitments_vec[0].empty());

  F expected_beta = F::FromHexString(
      "0x1a98a31b28f4089c8af7cdb48e9e056838679dffae6e96ff083cb8dba53a9143");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x2dc0d56a5db87e084ca72c1c51467b28d927ff94227126f04f69cb21965081f8");
  EXPECT_EQ(proof.gamma, expected_gamma);

  EXPECT_EQ(proof.permutation_product_commitments_vec.size(), num_circuits);
  EXPECT_TRUE(proof.permutation_product_commitments_vec[0].empty());

  ASSERT_EQ(proof.lookup_product_commitments_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_product_commitments_vec[0].empty());

  Commitment expected_vanishing_random_poly_commitment;
  {
    expected_vanishing_random_poly_commitment = CreateCommitment(
        {"0x0000000000000000000000000000000000000000000000000000000000000001",
         "0x0000000000000000000000000000000000000000000000000000000000000002"});
  }
  EXPECT_EQ(proof.vanishing_random_poly_commitment,
            expected_vanishing_random_poly_commitment);

  F expected_y = F::FromHexString(
      "0x17e9a336d7b197716486c320c264d80a5eab37371602ad38a6feb22486894c17");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x2bb7f1559c415c224b0c0713b6f281c9c2b6658f3af0c2c649bb0d3eb9b48660",
         "0x0b4621657cbc27228adc9011edbd154e9b1b743a9901ce7587b5802c4b93e4a5"},
        {"0x2f91475bd7f6c955ab4534ed4d0462491373479089f96c3d1b753a760e92c92d",
         "0x0f2b30e0179380f1a9dda5cb59eb5ff6c0c093f39ff3129b09a8fb4493fcb719"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x14479ed68753bcc9dfd9c779bc02c61fefa774a94b9b9030c94383bdc4bf4c43");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1ed705dd3a2a2cd4f3d290a7e94a003f718a16cefadc73fd338c824243307e8b",
        "0x05d1c76013b57315880ceb509d8e9d57a4f217ddbb38c5845aa7c9018603925e",
        "0x22ecbe355c5eff89abf6ed8bcaeffd14479afbcfff23e9a0d96a4e10625dd187",
        "0x0f8e7e452184691e52fb0b97791cef38222663b56e5196930fb528d63f5e0499",
        "0x1d4e474510d7c2d95f83102d5cf53193dea158a1311b82e6f9fed2fd8dda999d",
        "0x0f4efb86d0ce8d738b4ceba1c598c83493d37278794d90cfac272c424401ca6c",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x21e44b72e881fcce3e05e5986b8791122b9102ce4b2ace24e63c5586daa65d10",
        "0x22ebb08a7065b9ad3af2f867cebdfbbf7beb5045dbb17350247f50f68b1bda8c",
        "0x260e7742d9f390947d3ff69eb62b9838c095e3737c9f4a87d4e7dcf1841f3c51",
        "0x0cfd89da414305e253d05c5f08d921b616ac89dd8dd8ff548704b17e19ee81ee",
        "0x1adf6a915735625304545f3681cb9144a476bc607b657d411058efa27aaed8e7",
        "0x0297ccb03dbd36b770a70b2dc04ae7ee5e1568955c61fecc66599dd19f9ea992",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x26a4a18d596dd5e447aa545eba9120b47e90fbaf2dae07b831c5d5c394f993d8",
        "0x20d4b37b2339d0acf678850e372e711f5b083ca4fdacfea3e8b8dec88a56a3cc",
    };
    expected_fixed_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.fixed_evals, expected_fixed_evals);

  F expected_vanishing_random_eval = F::FromHexString(
      "0x0000000000000000000000000000000000000000000000000000000000000001");
  EXPECT_EQ(proof.vanishing_random_eval, expected_vanishing_random_eval);

  EXPECT_TRUE(proof.common_permutation_evals.empty());

  ASSERT_EQ(proof.permutation_product_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.permutation_product_evals_vec[0].empty());

  ASSERT_EQ(proof.permutation_product_next_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.permutation_product_next_evals_vec[0].empty());

  ASSERT_EQ(proof.permutation_product_last_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.permutation_product_last_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_product_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_product_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_product_next_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_product_next_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_input_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_permuted_input_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_input_inv_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_permuted_input_inv_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_table_evals_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_permuted_table_evals_vec[0].empty());

  F expected_h_eval = F::FromHexString(
      "0x0cdce47f9c4b0dfea0fffc3e184b36a02517c68ecae1d32921c9050771799bd2");
  EXPECT_EQ(h_eval, expected_h_eval);
}

}  // namespace tachyon::zk::plonk::halo2

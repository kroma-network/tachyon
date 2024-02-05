#include "tachyon/zk/plonk/examples/fibonacci/fibonacci1_circuit.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci1_circuit_test_data.h"
#include "tachyon/zk/plonk/halo2/pinned_constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

class Fibonacci1CircuitTest : public CircuitTest {};

}  // namespace

TEST_F(Fibonacci1CircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  Fibonacci1Config<F> config =
      Fibonacci1Circuit<F, SimpleFloorPlanner>::Configure(constraint_system);
  std::array<AdviceColumnKey, 3> expected_advice = {
      AdviceColumnKey(0),
      AdviceColumnKey(1),
      AdviceColumnKey(2),
  };
  EXPECT_EQ(config.advice, expected_advice);
  EXPECT_EQ(config.instance, InstanceColumnKey(0));
  EXPECT_EQ(config.selector, Selector::Simple(0));

  halo2::PinnedConstraintSystem<F> pinned_constraint_system(constraint_system);
  EXPECT_EQ(fibonacci1::kPinnedConstraintSystem,
            base::ToRustDebugString(pinned_constraint_system));

  EXPECT_TRUE(constraint_system.selector_map().empty());
  EXPECT_TRUE(constraint_system.general_column_annotations().empty());
}

TEST_F(Fibonacci1CircuitTest, Synthesize) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));
  const Domain* domain = prover_->domain();

  ConstraintSystem<F> constraint_system;
  Fibonacci1Config config =
      Fibonacci1Circuit<F, SimpleFloorPlanner>::Configure(constraint_system);
  Assembly<RationalEvals> assembly =
      VerifyingKey<F, Commitment>::CreateAssembly<RationalEvals>(
          domain, constraint_system);

  Fibonacci1Circuit<F, SimpleFloorPlanner> circuit;
  typename Fibonacci1Circuit<F, SimpleFloorPlanner>::FloorPlanner floor_planner;
  floor_planner.Synthesize(&assembly, circuit, std::move(config),
                           constraint_system.constants());

  EXPECT_TRUE(assembly.fixed_columns().empty());

  std::vector<AnyColumnKey> expected_columns = {
      AdviceColumnKey(0),
      AdviceColumnKey(1),
      AdviceColumnKey(2),
      InstanceColumnKey(0),
  };
  EXPECT_EQ(assembly.permutation().columns(), expected_columns);

  const CycleStore& cycle_store = assembly.permutation().cycle_store();
  // clang-format off
  CycleStore::Table<Label> expected_mapping({
      {{3, 0}, {3, 1},  {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{0, 1}, {2, 0},  {2, 1},  {2, 2},  {2, 3},  {2, 4},  {2, 5},  {2, 6},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{0, 2}, {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},  {1, 7},  {3, 2},
       {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 0}, {1, 0},  {2, 7},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<Label> expected_aux({
      {{0, 0}, {1, 0},  {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{1, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{1, 1}, {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},  {2, 7},
       {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{0, 0}, {1, 0},  {2, 7},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<size_t> expected_sizes({
      {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  });
  // clang-format on
  EXPECT_EQ(cycle_store.mapping(), expected_mapping);
  EXPECT_EQ(cycle_store.aux(), expected_aux);
  EXPECT_EQ(cycle_store.sizes(), expected_sizes);

  // clang-format off
  std::vector<std::vector<bool>> expected_selectors = {
      { true,  true,  true,  true,  true,  true,  true,  true,
       false, false, false, false, false, false, false, false}};
  // clang-format on
  EXPECT_EQ(assembly.selectors(), expected_selectors);
  EXPECT_EQ(assembly.usable_rows(), base::Range<RowIndex>::Until(10));
}

TEST_F(Fibonacci1CircuitTest, LoadVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci1Circuit<F, SimpleFloorPlanner> circuit;

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  halo2::PinnedVerifyingKey pinned_vkey(prover_.get(), vkey);
  EXPECT_EQ(fibonacci1::kPinnedVerifyingKey,
            base::ToRustDebugString(pinned_vkey));

  F expected_transcript_repr = F::FromHexString(
      "0x0e149c09b16d13bdc8a09508e1dab4af7399ebe708e0fc37a7fd59d43974596f");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

TEST_F(Fibonacci1CircuitTest, LoadProvingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci1Circuit<F, SimpleFloorPlanner> circuit;

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
      }};
      // clang-format on
      expected_fixed_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.fixed_polys(), expected_fixed_polys);

    std::vector<Evals> expected_permutations_columns;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> evals = {{
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
      }};
      // clang-format on
      expected_permutations_columns = CreateColumns(evals);
    }
    EXPECT_EQ(pkey.permutation_proving_key().permutations(),
              expected_permutations_columns);

    std::vector<Poly> expected_permutations_polys;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> polys = {{
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
      }};
      // clang-format on
      expected_permutations_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.permutation_proving_key().polys(),
              expected_permutations_polys);
  }
}

TEST_F(Fibonacci1CircuitTest, CreateProof) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci1Circuit<F, SimpleFloorPlanner> circuit;
  std::vector<Fibonacci1Circuit<F, SimpleFloorPlanner>> circuits = {
      circuit, std::move(circuit)};

  F a(1);
  F b(1);
  F out(55);
  std::vector<F> instance_column = {std::move(a), std::move(b), std::move(out)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuit));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(fibonacci1::kExpectedProof),
                                      std::end(fibonacci1::kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(Fibonacci1CircuitTest, VerifyProof) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci1Circuit<F, SimpleFloorPlanner> circuit;

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(fibonacci1::kExpectedProof),
                                   std::end(fibonacci1::kExpectedProof));
  Verifier<PCS> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));

  F a(1);
  F b(1);
  F out(55);
  std::vector<F> instance_column = {std::move(a), std::move(b), std::move(out)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
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
        {"0x093a25a0c4849e4437d1600bde8e3faacfb92a3bf15c2e2f311fa87ccc9be590",
         "0x23f368976bd1e762f05c5efc8db4e7590d2300e60ef4a7bd898cc2f5bbd771e6"},
        {"0x19fe09a37a4a987f305c5200e178b2afcad66f25081f0cfb6ee5460bae05b30b",
         "0x0e106dd8def54c87c517f1eb20a56175822ce6e65f99a3a0655fc3e2e8a3e94b"},
        {"0x116d6d4f42dc437a1871a9699400bf84ef825391345aee90bbab4e6673063431",
         "0x2c2d516d5c04579cca6c30460e27130515680a99050770aabcdd8ec5855a9373"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));

    points = {
        {"0x093a25a0c4849e4437d1600bde8e3faacfb92a3bf15c2e2f311fa87ccc9be590",
         "0x23f368976bd1e762f05c5efc8db4e7590d2300e60ef4a7bd898cc2f5bbd771e6"},
        {"0x19fe09a37a4a987f305c5200e178b2afcad66f25081f0cfb6ee5460bae05b30b",
         "0x0e106dd8def54c87c517f1eb20a56175822ce6e65f99a3a0655fc3e2e8a3e94b"},
        {"0x116d6d4f42dc437a1871a9699400bf84ef825391345aee90bbab4e6673063431",
         "0x2c2d516d5c04579cca6c30460e27130515680a99050770aabcdd8ec5855a9373"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x0e147bc018ea0dae60978bd4c802f89281f85d9cefaeef97e8b1c59d058825ef");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_permuted_commitments_vec[i].empty());
  }

  F expected_beta = F::FromHexString(
      "0x00e8932e10163041ad7c218ae33a024e9bd27598bc694a390cb45feacf635f03");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x293b80e793dafce1f1db5e5d7f266c9f18f22a29e2d3cac7437c0eb02964d584");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x0aac0adc789b91b676adaa86c09f60e45a206b2f4a76c96d5b92eee99c40c342",
         "0x02d89be67f1f0a190db5ac710351156edd14bd4c1c8d971e0c189b24cc1fd25c"},
        {"0x2bba171dea2b906be86ec13eac99d161fc5910582e5a2688c24743f726cba7fa",
         "0x2a83885d5e5d33ce854487c7b3c4f660ea1a0397d506b3309e9d3725ab222a55"},
        {"0x21ba10946a73fc1695ba31904ce7c4f11905b52c94c63f18a5b0ad023ea45012",
         "0x046efe02c21b19217f8bc9347dcd14a45b7fa8a637e140c158db26c2a666feed"},
        {"0x2ae8207ece8abefa39f0489f267964d62be5a5a0bbe90e3acfc467b90ca31b03",
         "0x069f5af770fd98d51ab2d9870ad1320a849edd453e5018b423ee66592157b0da"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));

    points = {
        {"0x144a39ad99bd6e2d0973f7a1ad752ba15cc2d3c43c459f0f0b11cb30857df748",
         "0x0de06616abf1d127658e5abb001513408168702f80581e12d817d0f5dcb3a9c0"},
        {"0x05b15d9bfa764c71787254d796ed110d99a7d56bd8fa2489dc32fc6bbb5537c8",
         "0x2463a4ac48f4955fa0aa849200e950d14f85616ffffb51bf9af080d68f676b31"},
        {"0x101842e2d0da314d2ae2dcb9d8eb4c9454e6843ff9187091de76daffb8c328c9",
         "0x11356be411434aed21ed05ef34e4dd688a3794f311f35979c033cb55cb6bd8bc"},
        {"0x05c81fc84621aef9cf9b11d092aa2607e3a7c6cd2f5c0194b1e58b855848ce8a",
         "0x12466a17eb84900aa737801b9adf80ca99482e3e1045a78df5bc9777747e1710"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));
  }
  EXPECT_EQ(proof.permutation_product_commitments_vec,
            expected_permutation_product_commitments_vec);

  ASSERT_EQ(proof.lookup_product_commitments_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_product_commitments_vec[i].empty());
  }

  Commitment expected_vanishing_random_poly_commitment;
  {
    expected_vanishing_random_poly_commitment = CreateCommitment(
        {"0x0000000000000000000000000000000000000000000000000000000000000001",
         "0x0000000000000000000000000000000000000000000000000000000000000002"});
  }
  EXPECT_EQ(proof.vanishing_random_poly_commitment,
            expected_vanishing_random_poly_commitment);

  F expected_y = F::FromHexString(
      "0x287bf794a84ee4b639e643202679380ef8cab5bfd8b3b6b30df1645e170e79f5");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x09d28cc7fb878bf0f489dbc9d9f4ccb28e044b2f21ded1c0278a1c048e6f3733",
         "0x23612388ba8985a9ac2a94e8c79ae5d6ac1d97e5e48582d0c49625fe450766e5"},
        {"0x17e2617edd35c8e7d19c336ada456762fb62e7f384ceef1cdebc1e7fadfefc55",
         "0x061f2154c9097a1ad8f784c6bf043e59142944506ccddc2bbc0fc7c01c50ff36"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x2d91afabb171fe83cb29e3753f33a39afc4bbe033bc15d77b776d2b00ae85823");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0df06b431c2fb295081a02e0042df92d17641b5645c8df155b9942d5be91cebd",
        "0x2051854d78d70ef6689c684fc1a57502cbefe0aef9b3294f883d4936e04565ec",
        "0x27eb6b942a49d07afb2d76605f6de1f31a84f953fb5408122ee29666bbe02d22",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x0df06b431c2fb295081a02e0042df92d17641b5645c8df155b9942d5be91cebd",
        "0x2051854d78d70ef6689c684fc1a57502cbefe0aef9b3294f883d4936e04565ec",
        "0x27eb6b942a49d07afb2d76605f6de1f31a84f953fb5408122ee29666bbe02d22",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x2de3bd02c39ba05237220cfbd03f7026fe290e7a77d4388380f7c509b0ca1362",
    };
    expected_fixed_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.fixed_evals, expected_fixed_evals);

  F expected_vanishing_random_eval = F::FromHexString(
      "0x0000000000000000000000000000000000000000000000000000000000000001");
  EXPECT_EQ(proof.vanishing_random_eval, expected_vanishing_random_eval);

  std::vector<F> expected_common_permutation_evals;
  {
    std::vector<std::string_view> evals = {
        "0x214296489128592466c39afe9f08e62f206183a9839d3732ccaf6c3cfefd3d2a",
        "0x2d9d160a8426e7c2f83cfa118fe4336a540066890b69c98d67e42faa22c644de",
        "0x2db8cce6b76aa80dfb7b0165b142bc2e626e1086f27b8a24ed8e606f2eb68431",
        "0x11216174cb8d7975406ab5360e5679529dc9f40d7b7f814d4b1b9a65c53bf653",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0a29a0d12339451e322f41dd5dbfdbfdee64c6801724466809d18b4e344c6c96",
        "0x14b4268628cec320986abb30f8e642e7afa63108a220f747364582e3beb6d855",
        "0x0c051dfdee2005124c2e650c845c323348f63453e1046962834ef545e5c606e2",
        "0x17665a33a76bf57fc64b2e7068e980ca246088ee22ca6b8de41762628ad38413",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x2efb25bf96faa98d4c3c6b64f60afea22d17fc5dc62c6c5051ea28af0a1d7cfa",
        "0x1a78985f9effe1c061e35a2d1d09c52e406d130af33d3325c7b8ea2f7bf653ba",
        "0x20141b7dae09f80023cbbd8935bbea85a5b6f18c1995bddf625f6a12fd75972e",
        "0x2d69e52101d86ad1020886f058a3afce055b14f2bac9e09b728734ecc741f505",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0957d06647ec178d57b6bf852813e409b87733ce7eb77fc6a9f8baab3a8d76f0",
        "0x0fa4a9986afea05c2ca141a14e446483d300062d60c265e616f52e91554eec78",
        "0x0ac831d220c34bcdecebf6034df7948eeb0afcf3e75d9ff0f7d22e8d53633615",
        "0x16d362ca2d3ad9b1c4fc2d1f80b39bf5cc280d2475226c03808da19e779c7a46",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x2c86b65c7c17fa96cf2df5128ce1f5699411c0ebac9dd19a21a5bbaa0f3208d8",
        "0x0f843ea6becaac1c3bab4507125a9baef3358f0ec37fca2efe1d208d168ddfb3",
        "0x26d197f4bd2305b48fa55fd729ee0528b20cf39ff4f855e7eefea0ccb3343610",
        "0x062158721b9d78e65aa290d1a8e910d4f09cbcb9e331e33f9de5408e49f79503",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x105a280ce342c30bfee86303fb6444be55672f81aed7d8feb94d7367f0ee4f32",
        "0x1a06c6bb9065db27cbb256808d8c61aca6d0fe46e67a616503e662e6f94b2375",
        "0x010f885db56e94615ab040d031cc67596e522c87d646a4195b0f0d7d753c3cc4",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));

    evals = {
        "0x1b86a13fb33315839484efc8366f3307e6107ff7e18bd4341e9ff1cab54280f2",
        "0x2d5f4c55c7dbde1dd8b682aaa773ff6061e7612a38d1bf900d6ad6aba1a553fe",
        "0x0d4bef98f059069961ea8a0c01ea0addccc0611564ce785ee580034b643e232d",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_last_evals_vec,
            expected_permutation_product_last_evals_vec);

  ASSERT_EQ(proof.lookup_product_evals_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_product_evals_vec[i].empty());
  }

  ASSERT_EQ(proof.lookup_product_next_evals_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_product_next_evals_vec[i].empty());
  }

  ASSERT_EQ(proof.lookup_permuted_input_evals_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_permuted_input_evals_vec[i].empty());
  }

  ASSERT_EQ(proof.lookup_permuted_input_inv_evals_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_permuted_input_inv_evals_vec[i].empty());
  }

  ASSERT_EQ(proof.lookup_permuted_table_evals_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_permuted_table_evals_vec[i].empty());
  }
}

}  // namespace tachyon::zk::plonk::halo2

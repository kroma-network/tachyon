#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci2_circuit.h"
#include "tachyon/zk/plonk/examples/fibonacci/fibonacci2_circuit_test_data.h"
#include "tachyon/zk/plonk/halo2/pinned_constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_floor_planner.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

class Fibonacci2V1CircuitTest : public CircuitTest {};

}  // namespace

TEST_F(Fibonacci2V1CircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  Fibonacci2Config<F> config =
      Fibonacci2Circuit<F, V1FloorPlanner>::Configure(constraint_system);
  EXPECT_EQ(config.advice, AdviceColumnKey(0));
  EXPECT_EQ(config.instance, InstanceColumnKey(0));
  EXPECT_EQ(config.selector, Selector::Simple(0));

  halo2::PinnedConstraintSystem<F> pinned_constraint_system(constraint_system);
  EXPECT_EQ(fibonacci2_v1::kPinnedConstraintSystem,
            base::ToRustDebugString(pinned_constraint_system));

  EXPECT_TRUE(constraint_system.selector_map().empty());
  EXPECT_TRUE(constraint_system.general_column_annotations().empty());
}

TEST_F(Fibonacci2V1CircuitTest, Synthesize) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));
  const Domain* domain = prover_->domain();

  ConstraintSystem<F> constraint_system;
  Fibonacci2Config config =
      Fibonacci2Circuit<F, V1FloorPlanner>::Configure(constraint_system);
  Assembly<RationalEvals> assembly =
      VerifyingKey<F, Commitment>::CreateAssembly<RationalEvals>(
          domain, constraint_system);

  Fibonacci2Circuit<F, V1FloorPlanner> circuit;
  typename Fibonacci2Circuit<F, V1FloorPlanner>::FloorPlanner floor_planner;
  floor_planner.Synthesize(&assembly, circuit, std::move(config),
                           constraint_system.constants());

  EXPECT_TRUE(assembly.fixed_columns().empty());

  std::vector<AnyColumnKey> expected_columns = {
      AdviceColumnKey(0),
      InstanceColumnKey(0),
  };
  EXPECT_EQ(assembly.permutation().columns(), expected_columns);

  const CycleStore& cycle_store = assembly.permutation().cycle_store();
  // clang-format off
  CycleStore::Table<Label> expected_mapping({
      {{1, 0}, {1, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {1, 2}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{0, 0}, {0, 1},  {0, 9},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
  });
  CycleStore::Table<Label> expected_aux({
      {{0, 0}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{0, 0}, {0, 1},  {0, 9},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
  });
  CycleStore::Table<size_t> expected_sizes({
      {2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1},
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

TEST_F(Fibonacci2V1CircuitTest, LoadVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci2Circuit<F, V1FloorPlanner> circuit;

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  halo2::PinnedVerifyingKey pinned_vkey(prover_.get(), vkey);
  EXPECT_EQ(fibonacci2_v1::kPinnedVerifyingKey,
            base::ToRustDebugString(pinned_vkey));

  F expected_transcript_repr = F::FromHexString(
      "0x0b95ba00d9df3c7f61587edd2ada6ca715d62dc579333da7b80365838146a2a1");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

TEST_F(Fibonacci2V1CircuitTest, LoadProvingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci2Circuit<F, V1FloorPlanner> circuit;

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
      }};
      // clang-format on
      expected_permutations_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.permutation_proving_key().polys(),
              expected_permutations_polys);
  }
}

TEST_F(Fibonacci2V1CircuitTest, CreateProof) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci2Circuit<F, V1FloorPlanner> circuit;
  std::vector<Fibonacci2Circuit<F, V1FloorPlanner>> circuits = {
      circuit, std::move(circuit)};

  F a = F(1);
  F b = F(1);
  F out = F(55);
  std::vector<F> instance_column = {std::move(a), std::move(b), std::move(out)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuit));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(fibonacci2_v1::kExpectedProof),
                                      std::end(fibonacci2_v1::kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(Fibonacci2V1CircuitTest, VerifyProof) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  Fibonacci2Circuit<F, V1FloorPlanner> circuit;

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(fibonacci2_v1::kExpectedProof),
                                   std::end(fibonacci2_v1::kExpectedProof));
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
        {"0x298dd784e9755190273d84998a7ed1f263a2a8e35fcdb344d8218990cab5c366",
         "0x1d42911d6e4d1799ebf92175f0a9210e911ff8dce85cce34235704fb0fd3c6e3"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));

    points = {
        {"0x298dd784e9755190273d84998a7ed1f263a2a8e35fcdb344d8218990cab5c366",
         "0x1d42911d6e4d1799ebf92175f0a9210e911ff8dce85cce34235704fb0fd3c6e3"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x1b28757736609214ea9e32e10c911f1b96ae85f38925661ddbd996be266fed09");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), num_circuits);
  for (size_t i = 0; i < num_circuits; ++i) {
    EXPECT_TRUE(proof.lookup_permuted_commitments_vec[i].empty());
  }

  F expected_beta = F::FromHexString(
      "0x0602223cf23aa71e528439ab0e0b6d9a2d931e3a22eba1c870bbeddbadb14983");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x00ec0e0eba8b1497529775edac3483f2db08278734c770f45b196fa7133faee3");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x1fb543b93d9d5e0393ef1dcab61fae9ebbc36176b2d2af8dbf57334254ebfbc7",
         "0x0e0ee2f2e8a5b8d690bc3fc179c1baa941b1990539d8a4b0d0029cb9a85cf8ea"},
        {"0x0479097d02790bea1db9c09a55f9ff31fb4485e0ba34f6efc41fc23e7d284120",
         "0x2a351db0e33baf97b4bf16b9937a45bbfeabb04d74d5e06d096a9eca57406d55"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));

    points = {
        {"0x1587949a63c990558e38fdf656379fc61aa4c78614bca1dd3a13586ec4dd0899",
         "0x2c5c9cf5d5c2044a43db333083cb3d46a02ac2d2009697ee99fde79f2498bfa7"},
        {"0x28086f2e6c40a7232cb5be4988f8ef35a55d5d9380867f5d83f686c38b9f1c00",
         "0x180a44b0d51f8a1fbdf278a6d2694e6a08f5fb9890b43fac70433fc6f0b62ee5"},
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
      "0x11abab761694e36f90d075f2e4ca450bf11653b3607a676fb9d89621f78e8568");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x2aa428dd3d06de9f752cbd475a741fbed536fc2035098a4bd06e49a3f7d4ff81",
         "0x09a52eb0da623bb6b44e8d58be168483f7a8435700257d8d501827689aa6ee00"},
        {"0x2b9a67d87f29b12474c5c3877d7afe1bb1fb063019b0139817b9ad747f133769",
         "0x24ecbbdabe2de5d884d1fd99e6f7399f0ffce88a0af718d5ac00305aacfc09ff"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x1cd48fc4021ff60cc4fdf3d4c96109dba22b63fe21c501310c5b3c14f56ce9aa");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1079268ca083ed8ab6dfde4412a0fbb175ce983a075ba1f7a1bab80a1ce4c57d",
        "0x0b27f48266439b9f32181c7d81a796bb5037ff4166aa64c16958160284fe41dd",
        "0x21516b020c3caa8b0ab7a89d0fc3429aa1eafa291eebef33aa91dc8169a2e583",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x1079268ca083ed8ab6dfde4412a0fbb175ce983a075ba1f7a1bab80a1ce4c57d",
        "0x0b27f48266439b9f32181c7d81a796bb5037ff4166aa64c16958160284fe41dd",
        "0x21516b020c3caa8b0ab7a89d0fc3429aa1eafa291eebef33aa91dc8169a2e583",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x21275a68fce5f900f1b0c9526bc9ca6409b56fa21354b6063684d6bf8a975bd1",
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
        "0x06fc3665d38e1f3a45cd60178fb0c040a9872f659d241ce7cdaf24e18b6ac4a9",
        "0x11e07a9a66d07f15a6d7ef1c8cc8c70195d4d0f51e194ce21601c90ea11fef55",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x12688d2019f303492c7aff4ea13d3774fce1d20f8477ddbd8420200cd1f145bd",
        "0x2054563bf1b19f7c4bcbafc050a7217cc65ed7535163ab2153e0ceee6785c4bc",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x0b435fe1ae7553d6157b30fa5aa9d2783d96e6a6e674412e65fffa6f689c687a",
        "0x1b1091a6e19966b091c2c158e6a586a2b3d0c0362cc4d18e085436a9bf7d1bb8",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x25e365b97577153fd7507f395340f1784dc3c6f271717da286c1df47b2d8fbd9",
        "0x08c2ea97e3812e28faf7f65b988908c576119e5aef6c025af55ef0fbd4cd92d0",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x2459f235b1aa666415aa364552a7ab7cd4b8616f2ef428f44aeb3690bdf30bf4",
        "0x2246c1930ac575b347fc9a6be3b48e2b56fdc14622cdb0687203fa69cb2c0341",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1671e528325cc88bc58d0818beecc3d143da9545a35dec26e23da7683ed24d4a",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));

    evals = {
        "0x07bb050cfa1565c5a235e3d39e40c3022e3b7523b5f3d52c4e80e61ff23bed8b",
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

#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/base/halo2/halo2_prover_test.h"
#include "tachyon/zk/plonk/keys/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon::zk {

namespace {

class SimpleCircuitTest : public Halo2ProverTest {};

}  // namespace

TEST_F(SimpleCircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  FieldConfig<math::bn254::Fr> config =
      SimpleCircuit<math::bn254::Fr>::Configure(constraint_system);
  std::array<AdviceColumnKey, 2> expected_advice = {
      AdviceColumnKey(0),
      AdviceColumnKey(1),
  };
  EXPECT_EQ(config.advice(), expected_advice);
  EXPECT_EQ(config.instance(), InstanceColumnKey(0));
  EXPECT_EQ(config.s_mul(), Selector::Simple(0));
  EXPECT_EQ(constraint_system.num_fixed_columns(), 1);
  EXPECT_EQ(constraint_system.num_advice_columns(), 2);
  EXPECT_EQ(constraint_system.num_instance_columns(), 1);
  EXPECT_EQ(constraint_system.num_selectors(), 1);
  EXPECT_EQ(constraint_system.num_challenges(), 0);
  std::vector<Phase> expected_advice_column_phases = {kFirstPhase, kFirstPhase};
  EXPECT_EQ(constraint_system.advice_column_phases(),
            expected_advice_column_phases);
  EXPECT_TRUE(constraint_system.challenge_phases().empty());
  EXPECT_TRUE(constraint_system.selector_map().empty());
  std::vector<Gate<math::bn254::Fr>> expected_gates;
  std::vector<std::unique_ptr<Expression<math::bn254::Fr>>> polys;
  {
    std::unique_ptr<Expression<math::bn254::Fr>> poly =
        ExpressionFactory<math::bn254::Fr>::Product(
            ExpressionFactory<math::bn254::Fr>::Selector(config.s_mul()),
            ExpressionFactory<math::bn254::Fr>::Sum(
                ExpressionFactory<math::bn254::Fr>::Product(
                    ExpressionFactory<math::bn254::Fr>::Advice(
                        AdviceQuery(0, Rotation::Cur(), config.advice()[0])),
                    ExpressionFactory<math::bn254::Fr>::Advice(
                        AdviceQuery(1, Rotation::Cur(), config.advice()[1]))),
                ExpressionFactory<math::bn254::Fr>::Negated(
                    ExpressionFactory<math::bn254::Fr>::Advice(AdviceQuery(
                        2, Rotation::Next(), config.advice()[0])))));
    polys.push_back(std::move(poly));
  }
  expected_gates.push_back(Gate<math::bn254::Fr>(
      "mul", {""}, std::move(polys), {Selector::Simple(0)},
      {
          {AdviceColumnKey(0), Rotation::Cur()},
          {AdviceColumnKey(1), Rotation::Cur()},
          {AdviceColumnKey(0), Rotation::Next()},
      }));
  EXPECT_EQ(constraint_system.gates(), expected_gates);
  std::vector<AdviceQueryData> expected_advice_queries = {
      AdviceQueryData(Rotation::Cur(), AdviceColumnKey(0)),
      AdviceQueryData(Rotation::Cur(), AdviceColumnKey(1)),
      AdviceQueryData(Rotation::Next(), AdviceColumnKey(0)),
  };
  EXPECT_EQ(constraint_system.advice_queries(), expected_advice_queries);
  std::vector<size_t> expected_num_advice_queries = {2, 1};
  EXPECT_EQ(constraint_system.num_advice_queries(),
            expected_num_advice_queries);
  std::vector<InstanceQueryData> expected_instance_queries = {
      InstanceQueryData(Rotation::Cur(), InstanceColumnKey(0)),
  };
  EXPECT_EQ(constraint_system.instance_queries(), expected_instance_queries);
  std::vector<FixedQueryData> expected_fixed_queries = {
      FixedQueryData(Rotation::Cur(), FixedColumnKey(0)),
  };
  EXPECT_EQ(constraint_system.fixed_queries(), expected_fixed_queries);
  std::vector<AnyColumnKey> expected_permutation_columns = {
      InstanceColumnKey(0),
      FixedColumnKey(0),
      AdviceColumnKey(0),
      AdviceColumnKey(1),
  };
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);
  EXPECT_TRUE(constraint_system.lookups().empty());
  EXPECT_TRUE(constraint_system.general_column_annotations().empty());
  EXPECT_FALSE(constraint_system.minimum_degree().has_value());
}

TEST_F(SimpleCircuitTest, Synthesize) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  ConstraintSystem<F> constraint_system;
  FieldConfig<math::bn254::Fr> config =
      SimpleCircuit<math::bn254::Fr>::Configure(constraint_system);
  Assembly<PCS> assembly =
      VerifyingKey<PCS>::CreateAssembly(prover_->pcs(), constraint_system);

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<math::bn254::Fr> circuit(constant, a, b);
  SimpleCircuit<math::bn254::Fr>::FloorPlanner::Synthesize(
      &assembly, circuit, std::move(config), constraint_system.constants());

  EXPECT_EQ(assembly.k(), 4);
  std::vector<RationalEvals> expected_fixed_columns;
  RationalEvals evals = RationalEvals::UnsafeZero(n - 1);
  *evals[0] = math::RationalField<F>(constant);
  expected_fixed_columns.push_back(std::move(evals));
  EXPECT_EQ(assembly.fixed_columns(), expected_fixed_columns);

  std::vector<AnyColumnKey> expected_columns = {
      InstanceColumnKey(0),
      FixedColumnKey(0),
      AdviceColumnKey(0),
      AdviceColumnKey(1),
  };
  EXPECT_EQ(assembly.permutation().columns(), expected_columns);

  const CycleStore& cycle_store = assembly.permutation().cycle_store();
  // clang-format off
  CycleStore::Table<Label> expected_mapping({
      {{2, 8}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 2}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 3}, {3, 3},  {2, 7},  {2, 0},  {3, 5},  {2, 4},  {3, 7},  {1, 0},
       {0, 0}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {3, 2},  {2, 1},  {3, 4},  {2, 5},  {3, 6},  {2, 6},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<Label> expected_aux({
      {{2, 8}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{1, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 3}, {3, 3},  {1, 0},  {2, 3},  {2, 5},  {2, 5},  {3, 7},  {1, 0},
       {2, 8}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {3, 2},  {3, 3},  {3, 4},  {2, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<size_t> expected_sizes({
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1},
  });
  // clang-format on
  EXPECT_EQ(cycle_store.mapping(), expected_mapping);
  EXPECT_EQ(cycle_store.aux(), expected_aux);
  EXPECT_EQ(cycle_store.sizes(), expected_sizes);

  // clang-format off
  std::vector<std::vector<bool>> expected_selectors = {
      {false, false, false,  true, false,  true, false,  true,
       false, false, false, false, false, false, false, false}};
  // clang-format on
  EXPECT_EQ(assembly.selectors(), expected_selectors);
  EXPECT_EQ(assembly.usable_rows(), base::Range<size_t>::Until(10));
}

TEST_F(SimpleCircuitTest, GenerateVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<math::bn254::Fr> circuit(constant, a, b);

  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(VerifyingKey<PCS>::Generate(prover_.get(), circuit, &vkey));

  struct Point {
    std::string_view x;
    std::string_view y;
  };
  std::vector<Commitment> expected_permutation_verifying_key;
  {
    std::vector<Point> points = {
        {"0x0365b8986f1c38476aa6479eea1b688244b4070413eb393efa5b06441ac2aeaa",
         "0x303ed0aaed99cb2848d844239ab4ab9a9191b86544edab860c42a2d4e504cf34"},
        {"0x1af13f7dc79a97c1a690215e2ffd3d8386f682fe1a13bdb8588d2fbaa0161edc",
         "0x10f0c0ddf9db782e88deb3e49464292b1276b2b9ad29b5a25971c85fe0b8aa96"},
        {"0x09837455c613e5b0e0edd2a1d47f1efdee21eeeda00548f3686c733301e23da4",
         "0x2e2b7776741eb2214916eb80ae0982c32b02e57e83c51e8907b5ffeab048bf2d"},
        {"0x23e033260e2f2ed7dd27295a06d951fad7c1545a493f24e8d7f416ccc3c74670",
         "0x0957c9d3b0c00ff783bc3357dd0b5cf891f49bc78f569126ba6e68bc394859ef"},
    };
    expected_permutation_verifying_key = base::Map(points, [](const Point& p) {
      return Commitment(math::bn254::Fq::FromHexString(p.x),
                        math::bn254::Fq::FromHexString(p.y));
    });
  }
  EXPECT_EQ(vkey.permutation_verifying_key().commitments(),
            expected_permutation_verifying_key);

  std::vector<Commitment> expected_fixed_commitments;
  {
    std::vector<Point> points = {
        {"0x0f9cc629d0010671d7f755267acccfc8d5854f47d4e437ec84bddcc27b9f19d1",
         "0x199b1dcc7aff518a4e49c6393bcf847cc3fde52ae69e326dea0d1d901424552a"},
        {"0x10472018b5bfdcc76f3925ea4f660dc7167ef12c96fb70f686d0c1cf7791cde5",
         "0x0358f44f7cb29a8d129dbe6b61fc1d921a903f1e9209abc137c7a6446c3ae38f"},
    };
    expected_fixed_commitments = base::Map(points, [](const Point& p) {
      return Commitment(math::bn254::Fq::FromHexString(p.x),
                        math::bn254::Fq::FromHexString(p.y));
    });
  }
  EXPECT_EQ(vkey.fixed_commitments(), expected_fixed_commitments);

  math::bn254::Fr expected_transcript_repr = math::bn254::Fr::FromHexString(
      "0x03b30e0717f2047e825763ccf9c91fff91c82eef5ec0834f66f359f29a3d3b58");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

}  // namespace tachyon::zk

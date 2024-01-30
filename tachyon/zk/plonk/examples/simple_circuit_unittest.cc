#include "tachyon/zk/plonk/examples/simple_circuit.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/zk/plonk/examples/circuit_test.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/simple_floor_planner.h"

namespace tachyon::zk::halo2 {

namespace {

constexpr uint8_t kExpectedProof[] = {
    206, 109, 139, 136, 181, 35,  204, 231, 212, 93,  105, 116, 154, 77,  204,
    23,  71,  148, 11,  151, 126, 145, 6,   150, 171, 185, 254, 230, 41,  136,
    76,  141, 132, 227, 154, 206, 134, 35,  253, 67,  8,   186, 228, 143, 116,
    139, 145, 119, 85,  253, 127, 208, 95,  153, 195, 112, 209, 116, 172, 45,
    15,  175, 128, 142, 204, 179, 55,  39,  34,  51,  72,  104, 203, 18,  200,
    167, 238, 128, 150, 95,  51,  70,  102, 245, 31,  126, 175, 75,  128, 131,
    210, 183, 26,  150, 167, 148, 4,   122, 209, 122, 247, 23,  28,  107, 152,
    91,  100, 24,  117, 196, 95,  56,  57,  63,  0,   13,  164, 147, 133, 185,
    227, 117, 218, 126, 171, 80,  126, 29,  105, 149, 45,  14,  144, 234, 250,
    146, 228, 251, 88,  94,  78,  192, 70,  209, 240, 185, 175, 207, 176, 237,
    223, 162, 182, 167, 55,  27,  174, 75,  86,  146, 242, 122, 52,  24,  231,
    152, 166, 17,  135, 33,  51,  216, 81,  14,  114, 175, 246, 221, 85,  47,
    12,  246, 175, 152, 25,  30,  71,  14,  217, 13,  253, 129, 1,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   193,
    124, 38,  97,  203, 45,  175, 171, 18,  114, 71,  19,  96,  21,  168, 135,
    110, 37,  229, 152, 175, 52,  149, 169, 112, 55,  141, 136, 233, 197, 87,
    33,  12,  248, 45,  245, 255, 25,  21,  60,  244, 81,  90,  111, 175, 128,
    152, 67,  107, 251, 200, 109, 221, 63,  39,  110, 145, 162, 222, 19,  222,
    231, 116, 159, 18,  176, 226, 254, 178, 169, 131, 246, 249, 216, 204, 245,
    126, 14,  4,   60,  243, 190, 16,  143, 233, 121, 48,  35,  89,  249, 30,
    177, 21,  176, 212, 15,  104, 131, 160, 212, 128, 253, 250, 162, 45,  82,
    102, 102, 122, 91,  166, 246, 215, 246, 183, 243, 105, 62,  217, 187, 0,
    204, 214, 174, 4,   59,  14,  38,  128, 53,  79,  244, 40,  240, 226, 63,
    154, 124, 81,  250, 85,  206, 241, 220, 188, 207, 0,   8,   38,  48,  13,
    222, 204, 177, 10,  132, 98,  177, 91,  38,  92,  92,  44,  89,  83,  82,
    255, 169, 79,  241, 50,  24,  75,  236, 80,  251, 125, 22,  180, 97,  200,
    127, 13,  206, 115, 245, 155, 160, 230, 136, 154, 30,  172, 120, 173, 39,
    72,  19,  73,  8,   116, 13,  98,  133, 200, 250, 234, 55,  132, 198, 99,
    242, 169, 149, 17,  203, 131, 253, 90,  251, 56,  205, 125, 9,   1,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    101, 219, 98,  123, 150, 137, 140, 239, 244, 207, 227, 226, 239, 50,  245,
    206, 4,   135, 3,   203, 197, 107, 95,  238, 229, 70,  60,  3,   84,  16,
    193, 21,  186, 141, 39,  27,  61,  37,  131, 39,  58,  247, 149, 128, 72,
    154, 7,   98,  194, 213, 81,  39,  101, 190, 142, 121, 16,  198, 34,  24,
    115, 198, 18,  26,  144, 125, 33,  75,  113, 228, 149, 7,   80,  26,  242,
    236, 2,   250, 37,  139, 215, 91,  203, 46,  222, 13,  30,  16,  43,  95,
    137, 62,  35,  128, 253, 0,   252, 149, 54,  172, 100, 182, 3,   142, 64,
    61,  20,  112, 162, 3,   168, 57,  60,  208, 203, 114, 237, 54,  238, 243,
    178, 83,  100, 252, 146, 17,  48,  19,  248, 244, 246, 126, 158, 218, 45,
    239, 52,  234, 165, 93,  254, 40,  74,  97,  79,  71,  103, 94,  68,  40,
    254, 199, 137, 112, 217, 4,   174, 90,  36,  10,  160, 10,  23,  50,  127,
    108, 35,  107, 233, 162, 124, 252, 90,  154, 69,  130, 138, 146, 152, 92,
    204, 222, 218, 175, 136, 170, 222, 10,  121, 198, 196, 0,   118, 232, 193,
    37,  167, 252, 236, 132, 180, 32,  133, 173, 84,  210, 183, 226, 17,  152,
    254, 134, 173, 241, 28,  87,  229, 48,  185, 197, 202, 126, 116, 13,  158,
    17,  242, 206, 155, 216, 163, 131, 207, 97,  146, 105, 198, 21,  29,  235,
    163, 253, 248, 54,  166, 140, 84,  109, 77,  88,  15,  133, 239, 177, 66,
    35,  141, 193, 65,  27,  252, 207, 0,   238, 201, 89,  146, 26,  31,  49,
    36,  248, 237, 114, 159, 174, 81,  102, 131, 204, 164, 209, 155, 208, 54,
    242, 11,  26,  18,  81,  255, 169, 79,  127, 254, 205, 201, 84,  81,  26,
    100, 113, 181, 175, 12,  64,  206, 141, 30,  234, 221, 69,  88,  123, 58,
    46,  251, 89,  163, 12,  0,   129, 169, 250, 211, 36,  4,   21,  235, 7,
    141, 16,  88,  60,  29,  62,  39,  190, 103, 126, 80,  118, 119, 139, 28,
    111, 2,   47,  160, 211, 93,  37,  106, 199, 117, 42,  209, 3,   67,  40,
    228, 53,  162, 184, 195, 168, 100, 242, 224, 72,  249, 239, 186, 235, 197,
    16,  140, 187, 204, 147, 92,  10,  4,   3,   195, 25,  12,  118, 126, 116,
    71,  45,  243, 105, 220, 34,  185, 219, 254, 31,  237, 109, 158, 176, 236,
    151, 52,  245, 254, 108, 56,  157, 136, 166, 107, 43,  100, 140, 44,  126,
    190, 248, 75,  134, 46,  182, 90,  241, 5,   37,  202, 235, 23,  237, 68,
    131, 22,  85,  163, 159, 155, 92,  163, 70,  168, 174, 112, 0,   188, 235,
    127, 169, 179, 70,  81,  52,  243, 101, 53,  74,  114, 117, 219, 241, 37,
    65,  109, 119, 112, 207, 45,  130, 99,  31,  149, 218, 238, 224, 161, 30,
    37,  157, 153, 114, 39,  20,  158, 233, 47,  4,   162, 237, 192, 218, 12,
    61,  155, 149, 69,  29,  211, 217, 53,  3,   249, 233, 19,  147, 126, 73,
    150, 24,  76,  186, 238, 245, 143, 157, 17,  150, 41,  114, 91,  230, 104,
    213, 121, 49,  182, 91,  207, 101, 207, 228, 155, 97,  73,  192, 230, 142,
    99,  202, 97,  38};

class SimpleCircuitTest : public CircuitTest {};

}  // namespace

TEST_F(SimpleCircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  FieldConfig<F> config =
      SimpleCircuit<F, SimpleFloorPlanner>::Configure(constraint_system);
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
  std::vector<Gate<F>> expected_gates;
  std::vector<std::unique_ptr<Expression<F>>> polys;
  {
    std::unique_ptr<Expression<F>> poly = ExpressionFactory<F>::Product(
        ExpressionFactory<F>::Selector(config.s_mul()),
        ExpressionFactory<F>::Sum(
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Advice(
                    AdviceQuery(0, Rotation::Cur(), config.advice()[0])),
                ExpressionFactory<F>::Advice(
                    AdviceQuery(1, Rotation::Cur(), config.advice()[1]))),
            ExpressionFactory<F>::Negated(ExpressionFactory<F>::Advice(
                AdviceQuery(2, Rotation::Next(), config.advice()[0])))));
    polys.push_back(std::move(poly));
  }
  expected_gates.push_back(Gate<F>("mul", {""}, std::move(polys),
                                   {Selector::Simple(0)},
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
  std::vector<RowIndex> expected_num_advice_queries = {2, 1};
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
  const Domain* domain = prover_->domain();

  ConstraintSystem<F> constraint_system;
  FieldConfig<F> config =
      SimpleCircuit<F, SimpleFloorPlanner>::Configure(constraint_system);
  Assembly<RationalEvals> assembly =
      VerifyingKey<PCS>::CreateAssembly<RationalEvals>(domain,
                                                       constraint_system);

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);
  typename SimpleCircuit<F, SimpleFloorPlanner>::FloorPlanner floor_planner;
  floor_planner.Synthesize(&assembly, circuit, std::move(config),
                           constraint_system.constants());

  std::vector<RationalEvals> expected_fixed_columns;
  RationalEvals evals = domain->Empty<RationalEvals>();
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
  EXPECT_EQ(assembly.usable_rows(), base::Range<RowIndex>::Until(10));
}

TEST_F(SimpleCircuitTest, LoadVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);

  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

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
    expected_permutation_verifying_key = CreateCommitments(points);
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
    expected_fixed_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(vkey.fixed_commitments(), expected_fixed_commitments);

  F expected_transcript_repr = F::FromHexString(
      "0x03b30e0717f2047e825763ccf9c91fff91c82eef5ec0834f66f359f29a3d3b58");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

TEST_F(SimpleCircuitTest, LoadProvingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);

  for (size_t i = 0; i < 2; ++i) {
    ProvingKey<PCS> pkey;
    bool load_verifying_key = i == 0;
    SCOPED_TRACE(
        absl::Substitute("load_verifying_key: $0", load_verifying_key));
    if (load_verifying_key) {
      VerifyingKey<PCS> vkey;
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
      }};
      // clang-format on
      expected_fixed_columns = CreateColumns(evals);
    }
    EXPECT_EQ(pkey.fixed_columns(), expected_fixed_columns);

    std::vector<Poly> expected_fixed_polys;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> polys = {{
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
      }};
      // clang-format on
      expected_fixed_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.fixed_polys(), expected_fixed_polys);

    std::vector<Evals> expected_permutations_columns;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> evals = {{
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
      }};
      // clang-format on
      expected_permutations_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.permutation_proving_key().polys(),
              expected_permutations_polys);
  }
}

TEST_F(SimpleCircuitTest, CreateProof) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);
  std::vector<SimpleCircuit<F, SimpleFloorPlanner>> circuits = {
      std::move(circuit)};

  F c = constant * a.Square() * b.Square();
  std::vector<F> instance_column = {std::move(c)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      std::move(instance_columns)};

  ProvingKey<PCS> pkey;
  ASSERT_TRUE(pkey.Load(prover_.get(), circuit));
  prover_->CreateProof(pkey, std::move(instance_columns_vec), circuits);

  std::vector<uint8_t> proof = prover_->GetWriter()->buffer().owned_buffer();
  std::vector<uint8_t> expected_proof(std::begin(kExpectedProof),
                                      std::end(kExpectedProof));
  EXPECT_THAT(proof, testing::ContainerEq(expected_proof));
}

TEST_F(SimpleCircuitTest, Verify) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, SimpleFloorPlanner> circuit(constant, a, b);

  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(kExpectedProof),
                                   std::end(kExpectedProof));
  Verifier<PCS> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));
  F c = constant * a.Square() * b.Square();
  std::vector<F> instance_column = {std::move(c)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      std::move(instance_columns)};

  Proof<F, Commitment> proof;
  F h_eval;
  ASSERT_TRUE(verifier.VerifyProofForTesting(vkey, instance_columns_vec, &proof,
                                             &h_eval));

  std::vector<std::vector<Commitment>> expected_advice_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x0d4c8829e6feb9ab9606917e970b944717cc4d9a74695dd4e7cc23b5888b6dce",
         "0x03a99ef4660a95515763e072043119fcbf6d3f3b709af6bf05b5c8b4d815a775"},
        {"0x0e80af0f2dac74d170c3995fd07ffd5577918b748fe4ba0843fd2386ce9ae384",
         "0x058b31b773e7a0e22f1ef9d6bbcc154b3dfaec09ff6c78084c1ae5c150f6624d"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x12a46ce074901bb2ec3136e73969ba388a925ace4891d853aa071cabaf4589ce");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_permuted_commitments_vec[0].empty());

  F expected_beta = F::FromHexString(
      "0x1e2502cf4ba7d2e862c9432f546db6549f0073ff75bcce16ec6ba78c12a1d682");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x13ca5867ed47dd5ee525ced9e7c6c82907ee4b622d638bc5ec5c484e850d561b");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x14a7961ab7d283804baf7e1ff56646335f9680eea7c812cb684833222737b3cc",
         "0x1775df6698bfa7af48a83408e92cf0a132f20438bf9636100567960c6f26bb59"},
        {"0x1d7e50ab7eda75e3b98593a40d003f39385fc47518645b986b1c17f77ad17a04",
         "0x00ef3b17b03c469c18380ab7acc3c5d5befd60a46a1659631d50fd480416cab6"},
        {"0x12564bae1b37a7b6a2dfedb0cfafb9f0d146c04e5e58fbe492faea900e2d9569",
         "0x0cf1580112a8d2918afd4ebe3b021a2e3844712efb346c49029a077a67e06df1"},
        {"0x01fd0dd90e471e1998aff60c2f55ddf6af720e51d833218711a698e718347af2",
         "0x2cc11fb912c0f0e8fcd677d966361edf4c005813981ed9c20fd10184835556f7"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));
  }
  EXPECT_EQ(proof.permutation_product_commitments_vec,
            expected_permutation_product_commitments_vec);

  ASSERT_EQ(proof.lookup_product_commitments_vec.size(), 1);
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
      "0x1af9ee1bca5f25fb7746430586cc6d4d4cc9152d5e3251e9c77fe38f6c9178b1");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x2157c5e9888d3770a99534af98e5256e87a8156013477212abaf2dcb61267cc1",
         "0x2b04d3cec50250e35b5919da1cfe8142db0a636c0f29c6427fe713f52c1fff60"},
        {"0x1f74e7de13dea2916e273fdd6dc8fb6b439880af6f5a51f43c1519fff52df80c",
         "0x1fd85823b8d30be9fdde823da1f8c4c876bb874a236050c3d04032999ac56eb7"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x16b464904a5cd90ce14436a70fcf094ec288e227a6fe7ece36e9630b5a18d6cc");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0fd4b015b11ef959233079e98f10bef33c040e7ef5ccd8f9f683a9b2fee2b012",
        "0x260e3b04aed6cc00bbd93e69f3b7f6d7f6a65b7a6666522da2fafd80d4a08368",
        "0x265bb162840ab1ccde0d30260800cfbcdcf1ce55fa517c9a3fe2f028f44f3580",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x1e9a88e6a09bf573ce0d7fc861b4167dfb50ec4b1832f14fa9ff5253592c5c5c",
        "0x097dcd38fb5afd83cb1195a9f263c68437eafac885620d740849134827ad78ac",
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
        "0x15c11054033c46e5ee5f6bc5cb038704cef532efe2e3cff4ef8c89967b62db65",
        "0x1a12c6731822c610798ebe652751d5c262079a488095f73a2783253d1b278dba",
        "0x00fd80233e895f2b101e0dde2ecb5bd78b25fa02ecf21a500795e4714b217d90",
        "0x13301192fc6453b2f3ee36ed72cbd03c39a803a270143d408e03b664ac3695fc",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0a245aae04d97089c7fe28445e67474f614a28fe5da5ea34ef2dda9e7ef6f4f8",
        "0x2342b1ef850f584d6d548ca636f8fda3eb1d15c6699261cf83a3d89bcef2119e",
        "0x255dd3a02f026f1c8b7776507e67be273e1d3c58108d07eb150424d3faa98100",
        "0x0070aea846a35c9b9fa355168344ed17ebca2505f15ab62e864bf8be7e2c8c64",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x00c4c6790adeaa88afdadecc5c98928a82459a5afc7ca2e96b236c7f32170aa0",
        "0x1a0bf236d09bd1a4cc836651ae9f72edf824311f1a9259c9ee00cffc1b41c18d",
        "0x03040a5c93ccbb8c10c5ebbaeff948e0f264a8c3b8a235e4284303d12a75c76a",
        "0x1ea1e0eeda951f63822dcf70776d4125f1db75724a3565f3345146b3a97febbc",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0d747ecac5b930e5571cf1ad86fe9811e2b7d254ad8520b484ecfca725c1e876",
        "0x0ca359fb2e3a7b5845ddea1e8dce400cafb571641a5154c9cdfe7f4fa9ff5112",
        "0x2b6ba6889d386cfef53497ecb09e6ded1ffedbb922dc69f32d47747e760c19c3",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_last_evals_vec,
            expected_permutation_product_last_evals_vec);

  ASSERT_EQ(proof.lookup_product_evals_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_product_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_product_next_evals_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_product_next_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_input_evals_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_permuted_input_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_input_inv_evals_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_permuted_input_inv_evals_vec[0].empty());

  ASSERT_EQ(proof.lookup_permuted_table_evals_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_permuted_table_evals_vec[0].empty());

  F expected_h_eval = F::FromHexString(
      "0x0b91cc8fe9296c94157f8f2b226da12d1ef112be8e9e30c88b402e1b36bbab6e");
  EXPECT_EQ(h_eval, expected_h_eval);
}

}  // namespace tachyon::zk::halo2

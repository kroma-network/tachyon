#include "gtest/gtest.h"

#include "tachyon/zk/plonk/circuit/examples/circuit_test.h"
#include "tachyon/zk/plonk/circuit/examples/simple_circuit.h"
#include "tachyon/zk/plonk/circuit/floor_planner/v1/v1_floor_planner.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"

namespace tachyon::zk::halo2 {

namespace {

constexpr uint8_t kExpectedProof[] = {
    19,  29,  49,  75,  8,   191, 254, 126, 107, 125, 58,  193, 235, 208, 41,
    102, 196, 144, 108, 7,   165, 223, 43,  109, 53,  216, 237, 43,  184, 227,
    83,  9,   184, 235, 153, 217, 206, 69,  132, 165, 210, 205, 116, 237, 238,
    162, 49,  49,  204, 46,  223, 58,  73,  229, 128, 143, 213, 188, 0,   212,
    5,   85,  163, 5,   143, 118, 177, 63,  150, 170, 126, 253, 76,  171, 24,
    60,  255, 11,  51,  154, 165, 99,  4,   75,  197, 89,  80,  146, 214, 252,
    144, 126, 78,  83,  109, 152, 147, 12,  50,  121, 140, 246, 218, 64,  22,
    250, 214, 94,  91,  40,  210, 216, 158, 14,  248, 237, 139, 166, 164, 181,
    163, 173, 40,  206, 118, 144, 35,  172, 2,   199, 115, 80,  179, 46,  200,
    239, 204, 233, 228, 14,  165, 73,  46,  111, 195, 217, 200, 239, 216, 85,
    203, 71,  121, 165, 128, 173, 40,  197, 132, 172, 180, 102, 167, 151, 87,
    51,  153, 17,  51,  0,   91,  29,  131, 77,  119, 167, 59,  228, 0,   245,
    222, 41,  206, 245, 188, 174, 66,  94,  184, 86,  112, 21,  1,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,
    101, 235, 212, 23,  48,  250, 152, 227, 110, 221, 191, 145, 170, 116, 253,
    211, 235, 129, 35,  197, 14,  237, 53,  55,  30,  186, 52,  130, 141, 169,
    16,  187, 240, 210, 95,  164, 202, 254, 94,  61,  148, 173, 220, 97,  194,
    73,  183, 97,  255, 255, 181, 227, 212, 132, 162, 250, 191, 203, 3,   183,
    253, 244, 27,  18,  148, 179, 134, 155, 110, 103, 49,  35,  144, 67,  122,
    143, 172, 18,  19,  192, 164, 239, 8,   150, 216, 47,  108, 7,   233, 179,
    42,  243, 209, 198, 22,  229, 8,   223, 4,   206, 14,  125, 85,  220, 42,
    126, 165, 7,   92,  14,  174, 33,  226, 14,  15,  238, 119, 212, 126, 155,
    25,  99,  125, 135, 251, 48,  34,  57,  187, 165, 61,  205, 218, 221, 255,
    91,  229, 146, 157, 37,  132, 26,  139, 195, 21,  34,  140, 157, 2,   228,
    4,   139, 177, 21,  3,   14,  91,  109, 15,  198, 85,  184, 232, 226, 145,
    66,  157, 168, 49,  238, 59,  183, 221, 151, 195, 29,  54,  16,  18,  2,
    171, 248, 78,  163, 246, 51,  212, 223, 134, 120, 0,   110, 212, 77,  61,
    47,  245, 156, 134, 93,  149, 14,  49,  47,  161, 0,   180, 184, 107, 112,
    126, 155, 32,  206, 243, 100, 157, 221, 227, 109, 22,  247, 20,  1,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    117, 175, 214, 40,  42,  192, 219, 238, 177, 187, 91,  221, 196, 15,  23,
    108, 96,  170, 142, 117, 100, 174, 120, 172, 31,  19,  238, 7,   68,  25,
    218, 37,  215, 85,  227, 16,  177, 27,  195, 120, 89,  3,   171, 31,  3,
    144, 250, 9,   234, 108, 158, 214, 24,  51,  65,  57,  147, 66,  240, 63,
    115, 129, 103, 5,   178, 196, 94,  172, 3,   149, 62,  234, 187, 227, 204,
    125, 217, 6,   51,  135, 185, 72,  128, 247, 203, 184, 240, 36,  148, 41,
    61,  155, 205, 11,  25,  17,  247, 55,  232, 120, 145, 100, 204, 146, 29,
    251, 153, 90,  190, 100, 63,  33,  191, 41,  230, 19,  31,  205, 112, 131,
    224, 251, 3,   169, 148, 73,  132, 20,  206, 157, 63,  155, 125, 104, 141,
    139, 150, 227, 154, 20,  111, 74,  70,  162, 17,  228, 72,  143, 160, 242,
    65,  174, 69,  132, 221, 195, 24,  62,  138, 16,  93,  90,  170, 161, 134,
    60,  84,  7,   186, 217, 207, 23,  199, 226, 206, 164, 170, 203, 139, 58,
    45,  101, 83,  202, 7,   90,  141, 54,  40,  200, 53,  35,  238, 216, 218,
    218, 135, 153, 61,  213, 10,  239, 5,   110, 179, 30,  149, 159, 233, 117,
    73,  229, 202, 91,  224, 83,  233, 80,  128, 241, 86,  166, 107, 29,  101,
    8,   15,  149, 82,  32,  84,  215, 112, 25,  10,  77,  128, 13,  153, 142,
    221, 17,  187, 117, 194, 193, 71,  170, 20,  38,  4,   77,  56,  101, 129,
    10,  226, 115, 135, 86,  10,  89,  27,  110, 66,  121, 147, 118, 188, 10,
    246, 157, 239, 48,  136, 133, 111, 134, 117, 12,  239, 84,  33,  152, 5,
    255, 96,  25,  0,   227, 174, 120, 122, 242, 15,  41,  216, 250, 253, 7,
    132, 18,  168, 174, 94,  239, 21,  59,  88,  67,  69,  196, 58,  135, 246,
    255, 181, 30,  51,  40,  155, 39,  28,  30,  218, 134, 84,  218, 184, 178,
    99,  95,  231, 99,  226, 152, 53,  99,  149, 84,  82,  226, 34,  18,  134,
    207, 2,   59,  113, 111, 235, 33,  175, 23,  167, 104, 139, 95,  10,  91,
    129, 202, 23,  15,  238, 59,  152, 106, 60,  71,  186, 78,  26,  141, 87,
    46,  34,  47,  52,  227, 62,  184, 124, 15,  88,  154, 52,  202, 15,  52,
    105, 19,  145, 47,  93,  228, 55,  153, 93,  194, 207, 209, 203, 187, 134,
    92,  227, 43,  249, 223, 24,  117, 122, 194, 106, 17,  7,   26,  63,  110,
    149, 92,  169, 51,  41,  225, 31,  95,  58,  159, 194, 28,  208, 81,  99,
    80,  47,  233, 216, 216, 69,  83,  216, 172, 169, 159, 241, 9,   36,  103,
    82,  151, 131, 92,  187, 141, 225, 203, 90,  165, 2,   227, 142, 52,  90,
    27,  111, 89,  41,  232, 41,  45,  60,  87,  185, 91,  105, 220, 62,  42,
    35,  117, 140, 51,  123, 127, 144, 60,  4,   72,  122, 166, 246, 62,  184,
    87,  199, 178, 151, 103, 205, 96,  44,  119, 136, 154, 153, 221, 54,  13,
    118, 46,  198, 206, 169, 119, 173, 57,  28,  0,   72,  145, 178, 181, 13,
    54,  38,  16,  39,  221, 68,  99,  207, 54,  100, 159, 247, 48,  28,  230,
    117, 251, 231, 130};

class SimpleV1CircuitTest : public CircuitTest {};

}  // namespace

TEST_F(SimpleV1CircuitTest, Configure) {
  ConstraintSystem<F> constraint_system;
  FieldConfig<F> config =
      SimpleCircuit<F, V1FloorPlanner>::Configure(constraint_system);
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

TEST_F(SimpleV1CircuitTest, Synthesize) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));
  const Domain* domain = prover_->domain();

  ConstraintSystem<F> constraint_system;
  FieldConfig<F> config =
      SimpleCircuit<F, V1FloorPlanner>::Configure(constraint_system);
  Assembly<PCS> assembly =
      VerifyingKey<PCS>::CreateAssembly(domain, constraint_system);

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, V1FloorPlanner> circuit(constant, a, b);
  typename SimpleCircuit<F, V1FloorPlanner>::FloorPlanner floor_planner;
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
      {{2, 1}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 6}, {0, 0},  {2, 5},  {3, 0},  {2, 8},  {3, 2},  {1, 0},  {3, 4},
       {2, 4}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{2, 3}, {3, 1},  {2, 2},  {3, 3},  {2, 7},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<Label> expected_aux({
      {{2, 1}, {0, 1},  {0, 2},  {0, 3},  {0, 4},  {0, 5},  {0, 6},  {0, 7},
       {0, 8}, {0, 9}, {0, 10}, {0, 11}, {0, 12}, {0, 13}, {0, 14}, {0, 15}},
      {{2, 0}, {1, 1},  {1, 2},  {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},
       {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}, {1, 13}, {1, 14}, {1, 15}},
      {{2, 0}, {2, 1},  {2, 2},  {3, 0},  {2, 4},  {2, 2},  {2, 0},  {3, 4},
       {2, 4}, {2, 9}, {2, 10}, {2, 11}, {2, 12}, {2, 13}, {2, 14}, {2, 15}},
      {{3, 0}, {3, 1},  {2, 2},  {3, 3},  {3, 4},  {3, 5},  {3, 6},  {3, 7},
       {3, 8}, {3, 9}, {3, 10}, {3, 11}, {3, 12}, {3, 13}, {3, 14}, {3, 15}},
  });
  CycleStore::Table<size_t> expected_sizes({
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {3, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  });
  // clang-format on
  EXPECT_EQ(cycle_store.mapping(), expected_mapping);
  EXPECT_EQ(cycle_store.aux(), expected_aux);
  EXPECT_EQ(cycle_store.sizes(), expected_sizes);

  // clang-format off
  std::vector<std::vector<bool>> expected_selectors = {
      {true, false, true, false, true, false, false, false,
       false, false, false, false, false, false, false, false}};
  // clang-format on
  EXPECT_EQ(assembly.selectors(), expected_selectors);
  EXPECT_EQ(assembly.usable_rows(), base::Range<size_t>::Until(10));
}

TEST_F(SimpleV1CircuitTest, LoadVerifyingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, V1FloorPlanner> circuit(constant, a, b);

  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<Commitment> expected_permutation_verifying_key;
  {
    std::vector<Point> points = {
        {"0x2f8d8133413e0224e1cbb20aa8281458fe0d9cf4d723ff07ac0524acffc8be48",
         "0x1f3e4dd4175e92cd38d3a8d3cd473e6daa6e9f28eb616222945904a6eadc52c6"},
        {"0x27f329c25618696f90d09ab41e80f4f2c1e927a6d0f86d8670e81882a449af36",
         "0x1a69f0c1a2dee36915704c4fdee41ff8b7029c1894dc89356dd2cf4738e95df2"},
        {"0x2a7f86208a87462b85678f6aee3c9c3ee17371167b436ad58501819ae02cbffd",
         "0x18b536eae002a69d57783fad5c63e35df0a2506dc43ee3d88053faff8f19cc40"},
        {"0x1bb17f6d2c12d9c8cd8860c00fcadebf63986bb330dfde5b9097b17047a4dfc1",
         "0x1d6be3858a788f18e3c16ca8d3a88893f3678be9f827463af0f30d6b1f7df112"},
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
        {"0x00253ad9f42498136d40e617f9d46d11bf33256914aab88e32fd413bed9baccc",
         "0x154f7ddc2d7959e1abc7a0cada48f1b8f079c9fb45c6fdc2ee2121cc83a83b16"},
    };
    expected_fixed_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(vkey.fixed_commitments(), expected_fixed_commitments);

  F expected_transcript_repr = F::FromHexString(
      "0x012577899026da8b4a257e25b4edf52711038e19083900a458e1c1e18c29eb08");
  EXPECT_EQ(vkey.transcript_repr(), expected_transcript_repr);
}

TEST_F(SimpleV1CircuitTest, LoadProvingKey) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, V1FloorPlanner> circuit(constant, a, b);

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
      }};
      // clang-format on
      expected_fixed_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.fixed_polys(), expected_fixed_polys);

    std::vector<Evals> expected_permutations_columns;
    {
      // clang-format off
      std::vector<std::vector<std::string_view>> evals = {{
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
      }};
      // clang-format on
      expected_permutations_polys = CreatePolys(polys);
    }
    EXPECT_EQ(pkey.permutation_proving_key().polys(),
              expected_permutations_polys);
  }
}

TEST_F(SimpleV1CircuitTest, Verify) {
  size_t n = 16;
  CHECK(prover_->pcs().UnsafeSetup(n, F(2)));
  prover_->set_domain(Domain::Create(n));

  F constant(7);
  F a(2);
  F b(3);
  SimpleCircuit<F, V1FloorPlanner> circuit(constant, a, b);

  VerifyingKey<PCS> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(kExpectedProof),
                                   std::end(kExpectedProof));
  std::unique_ptr<Verifier<PCS>> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));
  F c = constant * a.Square() * b.Square();
  std::vector<F> instance_column = {std::move(c)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      std::move(instance_columns)};

  Proof<F, Commitment> proof;
  F h_eval;
  ASSERT_TRUE(verifier->VerifyProofForTesting(vkey, instance_columns_vec,
                                              &proof, &h_eval));

  std::vector<std::vector<Commitment>> expected_advice_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x0953e3b82bedd8356d2bdfa5076c90c46629d0ebc13a7d6b7efebf084b311d13",
         "0x0e1846fc46b7f84859cf41eabe2cfadf1c08f6be2df3fd75f47dd9945ecdea66"},
        {"0x05a35505d400bcd58f80e5493adf2ecc3131a2eeed74cdd2a58445ced999ebb8",
         "0x167f11ce9f3d3ebc24fe1a9d722ffdc7ee94a1734d544837868c7ccd7de960f0"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x1e59668b92c989bea8d2ba08d3d58af7a9c4b941a5b144a05c68ff533f68986a");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), 1);
  EXPECT_TRUE(proof.lookup_permuted_commitments_vec[0].empty());

  F expected_beta = F::FromHexString(
      "0x21b46fb50af72088662a64a58e8b819e450b6bda307950d549278d3756ba4880");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x22736699ddc8d40e896f188a133d8489b4501c591a44b3d8982b3256c4ecd9ae");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x186d534e7e90fcd6925059c54b0463a59a330bff3c18ab4cfd7eaa963fb1768f",
         "0x2b2cfeda225a408ce72d76115abb8600857e00ef1e411560a7d77f59e0a0f879"},
        {"0x2c239076ce28ada3b5a4a68bedf80e9ed8d2285b5ed6fa1640daf68c79320c93",
         "0x04ae04434454674f0f9a54377501fdaae86391e08172b9d00ef5e49c6318c625"},
        {"0x2c84c528ad80a57947cb55d8efc8d9c36f2e49a50ee4e9ccefc82eb35073c702",
         "0x2d1e3c651112f3710bdcc3f370e20f6da9055b4cd6908e72db3e2e1ae613cc49"},
        {"0x157056b85e42aebcf5ce29def500e43ba7774d831d5b00331199335797a766b4",
         "0x1546d10e6b1d8f160ec28b1d815378f00c4a65140774e6147ddd793e36302fe8"},
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
      "0x2c2186391d08b03bf166e5b682e624dbcdc2ca63e9190f0934ce7473c44fd7e2");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x10a98d8234ba1e3735ed0ec52381ebd3fd74aa91bfdd6ee398fa3017d4eb6502",
         "0x0287856e16f35d446494172326c1964dc3f9feef33d0c5a2e830dff9e11a4d70"},
        {"0x1bf4fdb703cbbffaa284d4e3b5ffff61b749c261dcad943d5efecaa45fd2f0bb",
         "0x013b7018e287bd61722080446c0cbc056d0df5fa410222b93d881a6b63775864"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x0c0d8beac4282953e1a0662ea3dca6e8acada8a09b9aa23cc09aef9b90a0dc90");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x16c6d1f32ab3e9076c2fd89608efa4c01312ac8f7a43902331676e9b86b39412",
        "0x2230fb877d63199b7ed477ee0f0ee221ae0e5c07a57e2adc557d0ece04df08e5",
        "0x0f6d5b0e0315b18b04e4029d8c2215c38b1a84259d92e55bffdddacd3da5bb39",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x007886dfd433f6a34ef8ab021210361dc397ddb73bee31a89d4291e2e8b855c6",
        "0x14f7166de3dd9d64f3ce209b7e706bb8b400a12f310e955d869cf52f3d4dd46e",
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
        "0x25da194407ee131fac78ae64758eaa606c170fc4dd5bbbb1eedbc02a28d6af75",
        "0x056781733ff0429339413318d69e6cea09fa90031fab035978c31bb110e355d7",
        "0x11190bcd9b3d299424f0b8cbf78048b9873306d97dcce3bbea3e9503ac5ec4b2",
        "0x14844994a903fbe08370cd1f13e629bf213f64be5a99fb1d92cc649178e837f7",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x108a3e18c3dd8445ae41f2a08f48e411a2464a6f149ae3968b8d687d9b3f9dce",
        "0x0a8165384d042614aa47c1c275bb11dd8e990d804d0a1970d7542052950f0865",
        "0x21eb6f713b02cf861222e2525495633598e263e75f63b2b8da5486da1e1c279b",
        "0x09f19fa9acd85345d8d8e92f506351d01cc29f3a5f1fe12933a95c956e3f1a07",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x2335c828368d5a07ca53652d3a8bcbaaa4cee2c717cfd9ba07543c86a1aa5a5d",
        "0x1960ff05982154ef0c75866f858830ef9df60abc769379426e1b590a568773e2",
        "0x0f7cb83ee3342f222e578d1a4eba473c6a983bee0f17ca815b0a5f8b68a717af",
        "0x2a3edc695bb9573c2d29e829596f1b5a348ee302a55acbe18dbb5c8397526724",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x1d6ba656f18050e953e05bcae54975e99f951eb36e05ef0ad53d9987dadad8ee",
        "0x28331eb5fff6873ac44543583b15ef5eaea8128407fdfad8290ff27a78aee300",
        "0x116ac27a7518dff92be35c86bbcbd1cfc25d9937e45d2f911369340fca349a58",
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
      "0x020452325cb00a51db7420a281da9c60516fd63e544c2ecfe855b6e2d1ac0be2");
  EXPECT_EQ(h_eval, expected_h_eval);
}

}  // namespace tachyon::zk::halo2

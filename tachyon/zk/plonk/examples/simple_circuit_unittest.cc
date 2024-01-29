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
      VerifyingKey<F, Commitment>::CreateAssembly<RationalEvals>(
          domain, constraint_system);

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

  VerifyingKey<F, Commitment> vkey;
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
      circuit, std::move(circuit)};

  F c = constant * a.Square() * b.Square();
  std::vector<F> instance_column = {std::move(c)};
  std::vector<Evals> instance_columns = {Evals(std::move(instance_column))};
  std::vector<std::vector<Evals>> instance_columns_vec = {
      instance_columns, std::move(instance_columns)};

  ProvingKey<Poly, Evals, Commitment> pkey;
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

  VerifyingKey<F, Commitment> vkey;
  ASSERT_TRUE(vkey.Load(prover_.get(), circuit));

  std::vector<uint8_t> owned_proof(std::begin(kExpectedProof),
                                   std::end(kExpectedProof));
  Verifier<PCS> verifier =
      CreateVerifier(CreateBufferWithProof(absl::MakeSpan(owned_proof)));
  F c = constant * a.Square() * b.Square();
  std::vector<F> instance_column = {std::move(c)};
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
        {"0x0d4c8829e6feb9ab9606917e970b944717cc4d9a74695dd4e7cc23b5888b6dce",
         "0x03a99ef4660a95515763e072043119fcbf6d3f3b709af6bf05b5c8b4d815a775"},
        {"0x0e80af0f2dac74d170c3995fd07ffd5577918b748fe4ba0843fd2386ce9ae384",
         "0x058b31b773e7a0e22f1ef9d6bbcc154b3dfaec09ff6c78084c1ae5c150f6624d"},
    };
    expected_advice_commitments_vec.push_back(CreateCommitments(points));

    points = {
        {"0x0d4c8829e6feb9ab9606917e970b944717cc4d9a74695dd4e7cc23b5888b6dce",
         "0x03a99ef4660a95515763e072043119fcbf6d3f3b709af6bf05b5c8b4d815a775"},
        {"0x0e80af0f2dac74d170c3995fd07ffd5577918b748fe4ba0843fd2386ce9ae384",
         "0x058b31b773e7a0e22f1ef9d6bbcc154b3dfaec09ff6c78084c1ae5c150f6624d"}};
    expected_advice_commitments_vec.push_back(CreateCommitments(points));
  }
  EXPECT_EQ(proof.advices_commitments_vec, expected_advice_commitments_vec);

  EXPECT_TRUE(proof.challenges.empty());

  F expected_theta = F::FromHexString(
      "0x2b059aeea380dae6d29c1f1709cddfe7a7fa5d239d2ed38f7c5e7db17349d035");
  EXPECT_EQ(proof.theta, expected_theta);

  ASSERT_EQ(proof.lookup_permuted_commitments_vec.size(), num_circuits);
  EXPECT_TRUE(proof.lookup_permuted_commitments_vec[0].empty());

  F expected_beta = F::FromHexString(
      "0x14b7ba74ee783acecf23166da4c9ca6f4fe63cdc66758cae396ce0804e87d509");
  EXPECT_EQ(proof.beta, expected_beta);

  F expected_gamma = F::FromHexString(
      "0x12cbf1b371cc5262a7e4a2eda2e1c3171f84b76f64183d5c90305b8b65efe947");
  EXPECT_EQ(proof.gamma, expected_gamma);

  std::vector<std::vector<Commitment>>
      expected_permutation_product_commitments_vec;
  {
    std::vector<Point> points = {
        {"0x292e5e86b8b65fd61d0e09be7ebac3a833fcc1c79d59cd3ae33b246d0528e5f0",
         "0x1f266354b24e20bbae2fa25ffe113392c1b5df9ec5f2a2eb99f6cb5447f4fc24"},
        {"0x2ae9016e7d2d754e1774a5112a5a5bd3a5bd0a0eb2b09b700a2bee4ac0c29c6a",
         "0x15c499ad168f1446aaa20c9af669210de2ad0321c0b338110cbcaf0b1a331023"},
        {"0x1001294618f7779cf8ba6d0f37054d12a54ad841031e5cb9863400168fb587c8",
         "0x19f8d635ee9d52a4ad56ea25cfedaccb9e644bc89ef114036c23c137a6d6dc7a"},
        {"0x2631279da4e70ee6d80ba70bc54cc45676bcaa5283e470bf2d955d1888ae243e",
         "0x2085ebbe76f114b6287f8db2fbadeb9c88a72a94e13c2a280fa4a1823693f47c"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));

    points = {
        {"0x2fc754ca05a22f8b9ce57021209d843cfb9080cd00aecbe9aed917a3062ff0fb",
         "0x148f23b9c4ee958cdc46ee3bb7cb8f40b00e1f05c6944fbb166c87d4a34a64bc"},
        {"0x06ba4f8eaeea216bac7fce303594627bd20f7dadda0567c8e2d5c19e7a60621b",
         "0x1b9af694f8f7657a82b27d7376f29e723215a607780b20d9a6064938e58a3203"},
        {"0x04b1d19d52ae41435e3798fdfdf24bf65e3778199b3005a778ebf69f52e0f98e",
         "0x09bedea0070365f54dc7edffce4ddc96e83c74eaa8543e069f8ad7a56f5ef209"},
        {"0x05204633a068152c63a71f741e61511c5758077491fcf3e76f6cc981c7c2aa47",
         "0x01e44060b9b70dc9474317dc324a516f94878ade0ad8221ab46c5b0d5607cad4"},
    };
    expected_permutation_product_commitments_vec.push_back(
        CreateCommitments(points));
  }
  EXPECT_EQ(proof.permutation_product_commitments_vec,
            expected_permutation_product_commitments_vec);

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
      "0x2b635a1a9615e175bf98c02fcd0eca6a0196c48c2be713ad3c42089ede337d69");
  EXPECT_EQ(proof.y, expected_y);

  std::vector<Commitment> expected_vanishing_h_poly_commitments;
  {
    std::vector<Point> points = {
        {"0x1be0ccebc6f91d8bf9c3d65e4a17e71500a28baa5ef2499430a57db8f1922d9f",
         "0x0d6ebddee89b00c919f56501344541d27387bd6523b31e4aab73b885c7c70c16"},
        {"0x23e1be246ace0757bb94f486cc60c8004cf3d133981d9d8d6e2a9d64337e4e87",
         "0x277166abda65f12d816f6294fff5f0a258fc37025859ce3ea21a82315aa0d0ef"},
    };
    expected_vanishing_h_poly_commitments = CreateCommitments(points);
  }
  EXPECT_EQ(proof.vanishing_h_poly_commitments,
            expected_vanishing_h_poly_commitments);

  F expected_x = F::FromHexString(
      "0x1171a27273726e2363fc9167880d5cba29092819cb46c22594092c9a6bd3fc34");
  EXPECT_EQ(proof.x, expected_x);

  std::vector<std::vector<F>> expected_advice_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x239fb952abb75002e1e82b11d5b8f52e590c3181411fbfbccfb5f7db096dbfdd",
        "0x2e0eb718740d944f5f41d84d19055c96e971fe341eab4ceeb0b90296146b42bd",
        "0x100ee0f95bef707e0eed218d99872e2d7361d9ac92a43587ac9144e36df0e543",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x239fb952abb75002e1e82b11d5b8f52e590c3181411fbfbccfb5f7db096dbfdd",
        "0x2e0eb718740d944f5f41d84d19055c96e971fe341eab4ceeb0b90296146b42bd",
        "0x100ee0f95bef707e0eed218d99872e2d7361d9ac92a43587ac9144e36df0e543",
    };
    expected_advice_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.advice_evals_vec, expected_advice_evals_vec);

  std::vector<F> expected_fixed_evals;
  {
    std::vector<std::string_view> evals = {
        "0x192f455ac885e1adfa079e1f9585d71900acdafd585e18de3b0c826340e2cb24",
        "0x12231d031cbde7331c02483dc72cee093ef5c53a25685ae11e94da6d67132262",
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
        "0x28ce954039a64a9d9eaa7db92976a5751f4f2528b7af3975dfecfccb137ddcb9",
        "0x0b146a885edbdee9d098775d2b843b24041c03f7676d308c6544ad85aaa9f478",
        "0x2b057ecb16c7251ed6c4b3adb496fcad519b260f0dffbf987aea9f57b3cd6857",
        "0x18c1c415cda9e1a6b6e5adcf8becb4dacc50df1eee6e76a1a62b4a76d374f569",
    };
    expected_common_permutation_evals = CreateEvals(evals);
  }
  EXPECT_EQ(proof.common_permutation_evals, expected_common_permutation_evals);

  std::vector<std::vector<F>> expected_permutation_product_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x0e741458510b6262f2697880ee971153e2fecb45505f35521d3746747e822391",
        "0x072b53573e0ae56a2cf32f71210ea86dace261253e310e36fa4081fdf2fb3dc5",
        "0x298e64ed50591456264d5919f3780be7c6bbace9481b3dea82a1d027cfdd86ba",
        "0x1a398bd4be1af204cd59cb77442407234b90b9ff1bd148b8148910072647f4b3",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x09f72c99687994b60943b376fc45da29abf864c80798590843a4bf28a5d16007",
        "0x066715b256e40e4d5f0a96478498d94cd2579d9634f01a36cd41b317d91a6f15",
        "0x0497fa51634f560941bdcc199635bb665198d1837bb1f8f0c2e9d797b6dbc88f",
        "0x200957e69e8774a28cf981b08659867c715e98f95914b27360eb4164696d0e67",
    };
    expected_permutation_product_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_evals_vec,
            expected_permutation_product_evals_vec);

  std::vector<std::vector<F>> expected_permutation_product_next_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x22aa51a9929805c3dd79d9b3f0c2d1126b2193a52bfb9fa334cfda7e51c4f137",
        "0x01888deb91ed4e09a9f65f1090d6c54c8efcb87651e973e90c5c3ae9efc3065c",
        "0x10edd4965ebc5a5bec3d0fdef05fc1a9703aae12af64e3423e6d54d71f8a2088",
        "0x2aef582e2a148d76199507528560469b7170f4c5dd79617f8a407b7d748dfc06",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));

    evals = {
        "0x17dd15543245c129bb0c6622f3b40c64f1170bd311bea077001d1db6104eccda",
        "0x22c3f184afbbc1d098eff5184bc40fe62704d7fedd30ae27ae8993866ae7594a",
        "0x0a53b3c59370dea83a38316b91e46e8df5cb35ec7c3079ca732c2bcc8b48ac0b",
        "0x2e77bc6bac98cde4988659cf5f84da6c5053deb6700f9a532b4d161d939e19e3",
    };
    expected_permutation_product_next_evals_vec.push_back(CreateEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_next_evals_vec,
            expected_permutation_product_next_evals_vec);

  std::vector<std::vector<std::optional<F>>>
      expected_permutation_product_last_evals_vec;
  {
    std::vector<std::string_view> evals = {
        "0x2d9f72e0335a7cca749d4ded824e66056f12d57b30769d7c976d68b949726120",
        "0x16dc47a28ec8456e9873ae7915cbb9e233e25cd38177f9ac8b1cbe8cb1c053a5",
        "0x1d0dd164b8e022d6c43349ed8ece29719241bab1590daead55c885becab11b75",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));

    evals = {
        "0x1c64dc2990419e5c3b87d807f4c1f6dc56853b55c2b8423688785df06dafb7cc",
        "0x0973824dfa049695d9ce19a45527add4758b0524259d0b908a1b403611393726",
        "0x00abfec5c84e9f7bb9e1da8f84b008405de3f94e75b1161b17c4f7875233f080",
        "",
    };
    expected_permutation_product_last_evals_vec.push_back(
        CreateOptionalEvals(evals));
  }
  EXPECT_EQ(proof.permutation_product_last_evals_vec,
            expected_permutation_product_last_evals_vec);

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
      "0x1535a994ce99cdd4b9d17531a2925be1decc650d939dc63753f4fce93d78a5e4");
  EXPECT_EQ(h_eval, expected_h_eval);
}

}  // namespace tachyon::zk::halo2

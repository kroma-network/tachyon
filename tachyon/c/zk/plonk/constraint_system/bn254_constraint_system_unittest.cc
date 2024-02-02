#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/plonk/constraint_system/constraint_system.h"

namespace tachyon::zk::plonk {

namespace {

class ConstraintSystemTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::Fr::Init(); }

  void SetUp() override {
    cs_ = reinterpret_cast<tachyon_bn254_plonk_constraint_system*>(&cpp_cs_);
  }

 protected:
  ConstraintSystem<math::bn254::Fr> cpp_cs_;
  tachyon_bn254_plonk_constraint_system* cs_;
};

}  // namespace

TEST_F(ConstraintSystemTest, ComputeBlindingFactors) {
  EXPECT_EQ(tachyon_bn254_plonk_constraint_system_compute_blinding_factors(cs_),
            5);
}

TEST_F(ConstraintSystemTest, GetAdviceColumnPhases) {
  size_t phases_len;
  for (uint8_t i = 0; i < 3; ++i) {
    tachyon_bn254_plonk_constraint_system_get_advice_column_phases(cs_, nullptr,
                                                                   &phases_len);
    ASSERT_EQ(phases_len, i);

    cpp_cs_.CreateAdviceColumn(Phase(i));
  }

  tachyon_phase phases[3];
  tachyon_bn254_plonk_constraint_system_get_advice_column_phases(cs_, phases,
                                                                 &phases_len);
  ASSERT_EQ(phases_len, 3);
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(phases[i].value, i);
  }
}

TEST_F(ConstraintSystemTest, GetChallengePhases) {
  size_t phases_len;
  for (uint8_t i = 0; i < 3; ++i) {
    tachyon_bn254_plonk_constraint_system_get_challenge_phases(cs_, nullptr,
                                                               &phases_len);
    ASSERT_EQ(phases_len, i);

    cpp_cs_.CreateAdviceColumn(Phase(i));
    cpp_cs_.CreateChallengeUsableAfter(Phase(i));
  }

  tachyon_phase phases[3];
  tachyon_bn254_plonk_constraint_system_get_challenge_phases(cs_, phases,
                                                             &phases_len);
  ASSERT_EQ(phases_len, 3);
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(phases[i].value, i);
  }
}

TEST_F(ConstraintSystemTest, GetPhases) {
  size_t phases_len;
  for (uint8_t i = 0; i < 3; ++i) {
    tachyon_bn254_plonk_constraint_system_get_phases(cs_, nullptr, &phases_len);
    if (i == 0) {
      ASSERT_EQ(phases_len, 1);
    } else {
      ASSERT_EQ(phases_len, i);
    }

    cpp_cs_.CreateAdviceColumn(Phase(i));
  }

  tachyon_phase phases[3];
  tachyon_bn254_plonk_constraint_system_get_phases(cs_, phases, &phases_len);
  ASSERT_EQ(phases_len, 3);
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(phases[i].value, i);
  }
}

TEST_F(ConstraintSystemTest, GetNumFixedColumns) {
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_fixed_columns(cs_),
              i);
    cpp_cs_.CreateFixedColumn();
  }
}

TEST_F(ConstraintSystemTest, GetNumInstanceColumns) {
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(
        tachyon_bn254_plonk_constraint_system_get_num_instance_columns(cs_), i);
    cpp_cs_.CreateInstanceColumn();
  }
  EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_instance_columns(cs_),
            3);
}

TEST_F(ConstraintSystemTest, GetNumAdviceColumns) {
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_advice_columns(cs_),
              i);
    cpp_cs_.CreateAdviceColumn();
  }
  EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_advice_columns(cs_),
            3);
}

TEST_F(ConstraintSystemTest, GetNumChallenges) {
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_challenges(cs_), i);

    cpp_cs_.CreateAdviceColumn(Phase(i));
    cpp_cs_.CreateChallengeUsableAfter(Phase(i));
  }
  EXPECT_EQ(tachyon_bn254_plonk_constraint_system_get_num_challenges(cs_), 3);
}

TEST_F(ConstraintSystemTest, GetConstants) {
  size_t constants_len;
  for (uint8_t i = 0; i < 3; ++i) {
    tachyon_bn254_plonk_constraint_system_get_constants(cs_, nullptr,
                                                        &constants_len);
    ASSERT_EQ(constants_len, i);

    FixedColumnKey column = cpp_cs_.CreateFixedColumn();
    cpp_cs_.EnableConstant(column);
  }

  tachyon_fixed_column_key constants[3];
  tachyon_bn254_plonk_constraint_system_get_constants(cs_, constants,
                                                      &constants_len);
  ASSERT_EQ(constants_len, 3);
  for (uint8_t i = 0; i < 3; ++i) {
    EXPECT_EQ(constants[i].index, i);
  }
}

}  // namespace tachyon::zk::plonk

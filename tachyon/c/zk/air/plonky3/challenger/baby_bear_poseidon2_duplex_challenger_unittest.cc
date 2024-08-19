#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/plonky3/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::air::plonky3 {

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>>;

class DuplexChallengerTest : public math::FiniteFieldTest<F> {
 public:
  void SetUp() override {
    challenger_ =
        tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_create();
  }

  void TearDown() override {
    tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_destroy(challenger_);
  }

 protected:
  tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger_ = nullptr;
};

TEST_F(DuplexChallengerTest, APIs) {
  DuplexChallenger<Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
                   TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>
      challenger = c::base::native_cast(*challenger_);
  for (size_t i = 0; i < 20; ++i) {
    F value(i);
    challenger.Observe(value);

    tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_observe(
        challenger_, c::base::c_cast(&value));
  }
  for (size_t i = 0; i < 10; ++i) {
    tachyon_baby_bear value =
        tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_sample(
            challenger_);
    EXPECT_EQ(c::base::native_cast(value), challenger.Sample());
  }
}

}  // namespace tachyon::zk::air::plonky3

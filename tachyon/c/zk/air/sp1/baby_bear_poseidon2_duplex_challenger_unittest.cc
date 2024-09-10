#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::BabyBear;
using Params = tachyon::crypto::Poseidon2Params<
    F, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA>;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>>,
                    Params>;

class DuplexChallengerTest : public math::FiniteFieldTest<F> {
 public:
  void SetUp() override {
    challenger_ = tachyon_sp1_baby_bear_poseidon2_duplex_challenger_create();
  }

  void TearDown() override {
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(challenger_);
  }

 protected:
  tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger_ = nullptr;
};

TEST_F(DuplexChallengerTest, APIs) {
  DuplexChallenger<Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>
      challenger = c::base::native_cast(*challenger_);
  for (size_t i = 0; i < 20; ++i) {
    F value(i);
    challenger.Observe(value);

    tachyon_sp1_baby_bear_poseidon2_duplex_challenger_observe(
        challenger_, c::base::c_cast(&value));
  }
  for (size_t i = 0; i < 10; ++i) {
    tachyon_baby_bear value =
        tachyon_sp1_baby_bear_poseidon2_duplex_challenger_sample(challenger_);
    EXPECT_EQ(c::base::native_cast(value), challenger.Sample());
  }
}

}  // namespace tachyon::crypto

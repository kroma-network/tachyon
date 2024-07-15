#include "tachyon/c/crypto/random/rng.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/crypto/random/cha_cha20/cha_cha20_rng.h"
#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"

namespace tachyon::crypto {

template <typename Rng>
class RngTest : public testing::Test {
 public:
  void TearDown() override {
    tachyon_rng_destroy(rng_);
    tachyon_rng_destroy(rng_clone_);
  }

 protected:
  tachyon_rng* rng_ = nullptr;
  tachyon_rng* rng_clone_ = nullptr;
};

using RngTypes = testing::Types<XORShiftRNG, ChaCha20RNG>;
TYPED_TEST_SUITE(RngTest, RngTypes);

TYPED_TEST(RngTest, APIs) {
  using Rng = TypeParam;

  std::vector<uint8_t> seed = base::CreateVector(
      Rng::kSeedSize, []() { return base::Uniform(base::Range<uint8_t>()); });
  Rng cpp_rng;
  ASSERT_TRUE(cpp_rng.SetSeed(seed));

  uint8_t type;
  if constexpr (std::is_same_v<Rng, XORShiftRNG>) {
    type = TACHYON_RNG_XOR_SHIFT;
  } else if constexpr (std::is_same_v<Rng, ChaCha20RNG>) {
    type = TACHYON_RNG_CHA_CHA20;
  }

  this->rng_ = tachyon_rng_create_from_seed(type, seed.data(), Rng::kSeedSize);

  EXPECT_EQ(cpp_rng.NextUint32(), tachyon_rng_get_next_u32(this->rng_));
  EXPECT_EQ(cpp_rng.NextUint64(), tachyon_rng_get_next_u64(this->rng_));

  size_t state_len;
  tachyon_rng_get_state(this->rng_, nullptr, &state_len);
  ASSERT_EQ(state_len, Rng::kStateSize);

  uint8_t state[Rng::kStateSize];
  tachyon_rng_get_state(this->rng_, state, &state_len);
  ASSERT_EQ(state_len, Rng::kStateSize);

  this->rng_clone_ = tachyon_rng_create_from_state(type, state, state_len);

  EXPECT_EQ(tachyon_rng_get_next_u32(this->rng_),
            tachyon_rng_get_next_u32(this->rng_clone_));
}

}  // namespace tachyon::crypto

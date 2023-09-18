#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_util.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::crypto {

namespace {
template <typename PrimeFieldType>
class PoseidonTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
#if defined(TACHYON_GMP_BACKEND)
    if constexpr (std::is_same_v<PrimeFieldType, math::bls12_381::FrGmp>) {
      PrimeFieldType::Init();
    }
#endif  // defined(TACHYON_GMP_BACKEND)
  }
};
}  // namespace

using PrimeFieldTypes = testing::Types<math::bls12_381::FrConfig>;
TYPED_TEST_SUITE(PoseidonTest, PrimeFieldTypes);

TYPED_TEST(PoseidonTest, TestGetDefaultPoseidonParameters) {
  using Config = TypeParam;
  auto constraints_rate_2 = GetDefaultPoseidonParameters<Config>(2, false);

  // clang-format off
  EXPECT_EQ(constraints_rate_2.ark[0][0],
            math::bls12_381::Fr::FromDecString(
                "27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  // clang-format on
}
}  // namespace tachyon::crypto

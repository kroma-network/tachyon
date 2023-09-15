#include "gmock/gmock.h"
#include "gtest/gtest.h"
//clang-format off
#include "tachyon/crypto/hashes/sponge/poseidon/grain_lfsr.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"
// clang-format on

namespace tachyon::crypto {

namespace {
template <typename PrimeFieldType>
class PoseidonGrainLFSRTest : public testing::Test {
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
TYPED_TEST_SUITE(PoseidonGrainLFSRTest, PrimeFieldTypes);

TYPED_TEST(PoseidonGrainLFSRTest, TestGetBits) {
  using PrimeFieldTy = TypeParam;
  PoseidonGrainLFSR<math::PrimeField<PrimeFieldTy>> lfsr(true, 255, 3, 8, 31);

  auto bits = lfsr.GetBits(16);
  EXPECT_EQ(bits.size(), 16);
}

TYPED_TEST(PoseidonGrainLFSRTest, TestGetFieldElementsModP) {
  using PrimeFieldTy = TypeParam;
  PoseidonGrainLFSR<math::PrimeField<PrimeFieldTy>> lfsr(true, 255, 3, 8, 31);

  auto elements = lfsr.GetFieldElementsModP(10);
  ASSERT_EQ(elements.size(), 10);
}

TYPED_TEST(PoseidonGrainLFSRTest, TestGetFieldElementsRejectionSampling) {
  using PrimeFieldTy = TypeParam;
  PoseidonGrainLFSR<math::PrimeField<PrimeFieldTy>> lfsr(false, 255, 3, 8, 31);

  // clang-format off
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            math::bls12_381::Fr::FromDecString(
                "27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  EXPECT_EQ(lfsr.GetFieldElementsRejectionSampling(1)[0],
            math::bls12_381::Fr::FromDecString(
                "51641662388546346858987925410984003801092143452466182801674685248597955169158"));
  // clang-format on
}

}  // namespace tachyon::crypto

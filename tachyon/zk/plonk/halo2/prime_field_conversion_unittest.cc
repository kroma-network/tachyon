#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::plonk::halo2 {

namespace {

using F = tachyon::math::bn254::Fq;

class PrimeFieldConversionTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(PrimeFieldConversionTest, FromUint128) {
  uint64_t limbs4[4] = {
      0x0,
      0x1,
      0x0,
      0x0,
  };
  F expected{math::BigInt<4>(limbs4)};

  EXPECT_EQ(FromUint128<F>(absl::int128(1) << 64), expected);
}

TEST_F(PrimeFieldConversionTest, FromUint512) {
  uint64_t limbs4[4] = {
      0x1f8905a172affa8a,
      0xde45ad177dcf3306,
      0xaaa7987907d73ae2,
      0x24d349431d468e30,
  };
  F expected{math::BigInt<4>(limbs4)};

  uint64_t limbs8[8] = {0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa,
                        0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa,
                        0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa,
                        0xaaaaaaaaaaaaaaaa, 0xaaaaaaaaaaaaaaaa};
  EXPECT_EQ(FromUint512<F>(limbs8), expected);

  uint8_t bytes64[64] = {
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
      0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa};
  EXPECT_EQ(FromUint512<F>(bytes64), expected);
}

}  // namespace tachyon::zk::plonk::halo2

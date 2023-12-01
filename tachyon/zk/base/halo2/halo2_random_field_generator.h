#ifndef TACHYON_ZK_BASE_HALO2_HALO2_RANDOM_FIELD_GENERATOR_H_
#define TACHYON_ZK_BASE_HALO2_HALO2_RANDOM_FIELD_GENERATOR_H_

#include "gtest/gtest_prod.h"

#include "tachyon/crypto/random/xor_shift/xor_shift_rng.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/zk/base/random_field_generator.h"

namespace tachyon::zk {

template <typename F>
class Halo2RandomFieldGenerator : public RandomFieldGenerator<F> {
 public:
  static_assert(F::BigIntTy::kLimbNums == 4,
                "Halo2Curves seems only supporting ~256 bit prime field.");

  explicit Halo2RandomFieldGenerator(crypto::XORShiftRNG* generator)
      : generator_(generator) {}

  // RandomFieldGenerator<F> methods
  F Generate() override {
    uint64_t limbs[8] = {
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
    };
    return FromUint512(limbs);
  }

 private:
  FRIEND_TEST(Halo2RandomFieldGeneratorTest, FromUint512);

  // See
  // https://github.com/kroma-network/halo2curves/blob/c0ac1935e5da2a620204b5b011be2c924b1e0155/src/derive/field.rs#L29-L47.
  static F FromUint512(uint64_t limbs[8]) {
    F d0 = F::FromMontgomery(
        math::BigInt<4>({limbs[0], limbs[1], limbs[2], limbs[3]}));
    F d1 = F::FromMontgomery(
        math::BigInt<4>({limbs[4], limbs[5], limbs[6], limbs[7]}));
    // NOTE(chokobole): When performing d0 * F::Config::kMontgomeryR2 + d1 *
    // F::Config::kMontgomeryR3, the result may be incorrect. This is due to our
    // prime field multiplication, where we utilize unused modulus bits for
    // optimization purposes. However, the given |limbs| can sometimes exceed
    // the allowed scope of bits.
    math::BigInt<8> mul_result =
        d0.ToMontgomery().Mul(F::Config::kMontgomeryR2);
    math::BigInt<4> d2;
    math::BigInt<4>::MontgomeryReduce64<false>(mul_result, F::Config::kModulus,
                                               F::Config::kInverse64, &d2);
    math::BigInt<8> mul_result2 =
        d1.ToMontgomery().Mul(F::Config::kMontgomeryR3);
    math::BigInt<4> d3;
    math::BigInt<4>::MontgomeryReduce64<false>(mul_result2, F::Config::kModulus,
                                               F::Config::kInverse64, &d3);
    return F::FromMontgomery(d2) + F::FromMontgomery(d3);
  }

  crypto::XORShiftRNG* const generator_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_HALO2_HALO2_RANDOM_FIELD_GENERATOR_H_

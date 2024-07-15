#ifndef TACHYON_ZK_PLONK_HALO2_RANDOM_FIELD_GENERATOR_H_
#define TACHYON_ZK_PLONK_HALO2_RANDOM_FIELD_GENERATOR_H_

#include "tachyon/crypto/random/rng.h"
#include "tachyon/zk/base/random_field_generator_base.h"
#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"

namespace tachyon::zk::plonk::halo2 {

template <typename F>
class RandomFieldGenerator : public RandomFieldGeneratorBase<F> {
 public:
  static_assert(F::BigIntTy::kLimbNums == 4,
                "Halo2Curves seems only supporting ~256 bit prime field.");

  explicit RandomFieldGenerator(crypto::RNG* generator)
      : generator_(generator) {}

  // RandomFieldGeneratorBase<F> methods
  F Generate() override {
    uint64_t limbs[8] = {
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
        generator_->NextUint64(), generator_->NextUint64(),
    };
    return FromUint512<F>(limbs);
  }

 private:
  // not owned
  crypto::RNG* const generator_;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_RANDOM_FIELD_GENERATOR_H_

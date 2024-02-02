#include "vendors/halo2/include/bn254_rational_evals.h"

#include "vendors/halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

RationalEvals::RationalEvals()
    : evals_(tachyon_bn254_univariate_rational_evaluations_create()) {}

RationalEvals::~RationalEvals() {
  tachyon_bn254_univariate_rational_evaluations_destroy(evals_);
}

size_t RationalEvals::len() const {
  return tachyon_bn254_univariate_rational_evaluations_len(evals_);
}

void RationalEvals::set_zero(size_t idx) {
  tachyon_bn254_univariate_rational_evaluations_set_zero(evals_, idx);
}

void RationalEvals::set_trivial(size_t idx, const Fr& numerator) {
  tachyon_bn254_univariate_rational_evaluations_set_trivial(
      evals_, idx, reinterpret_cast<const tachyon_bn254_fr*>(&numerator));
}

void RationalEvals::set_rational(size_t idx, const Fr& numerator,
                                 const Fr& denominator) {
  tachyon_bn254_univariate_rational_evaluations_set_rational(
      evals_, idx, reinterpret_cast<const tachyon_bn254_fr*>(&numerator),
      reinterpret_cast<const tachyon_bn254_fr*>(&denominator));
}

std::unique_ptr<RationalEvals> RationalEvals::clone() const {
  return std::make_unique<RationalEvals>(
      tachyon_bn254_univariate_rational_evaluations_clone(evals_));
}

}  // namespace tachyon::halo2_api::bn254

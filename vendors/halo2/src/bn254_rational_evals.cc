#include "vendors/halo2/include/bn254_rational_evals.h"

#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_rational_evals_impl.h"

namespace tachyon::halo2_api::bn254 {

RationalEvals::RationalEvals() : impl_(new RationalEvalsImpl()) {}

size_t RationalEvals::len() const {
  return impl_->evals().evaluations().size();
}

void RationalEvals::set_zero(size_t idx) {
  impl_->evals().evaluations()[idx] =
      math::RationalField<math::bn254::Fr>::Zero();
}

void RationalEvals::set_trivial(size_t idx, const Fr& numerator) {
  impl_->evals().evaluations()[idx] = math::RationalField<math::bn254::Fr>(
      reinterpret_cast<const math::bn254::Fr&>(numerator));
}

void RationalEvals::set_rational(size_t idx, const Fr& numerator,
                                 const Fr& denominator) {
  impl_->evals().evaluations()[idx] = {
      reinterpret_cast<const math::bn254::Fr&>(numerator),
      reinterpret_cast<const math::bn254::Fr&>(denominator),
  };
}

std::unique_ptr<RationalEvals> RationalEvals::clone() const {
  std::unique_ptr<RationalEvals> ret(new RationalEvals);
  ret->impl()->evals() = impl_->evals();
  return ret;
}

}  // namespace tachyon::halo2_api::bn254

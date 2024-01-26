#include "vendors/halo2/include/bn254_evals.h"

#include "vendors/halo2/src/bn254.rs.h"
#include "vendors/halo2/src/bn254_evals_impl.h"

namespace tachyon::halo2_api::bn254 {

Evals::Evals() : impl_(new EvalsImpl()) {}

size_t Evals::len() const { return impl_->evals().evaluations().size(); }

void Evals::set_value(size_t idx, const Fr& fr) {
  impl_->evals().evaluations()[idx] =
      reinterpret_cast<const math::bn254::Fr&>(fr);
}

}  // namespace tachyon::halo2_api::bn254

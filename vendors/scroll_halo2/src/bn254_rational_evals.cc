#include "vendors/scroll_halo2/include/bn254_rational_evals.h"

#include "vendors/scroll_halo2/include/bn254_rational_evals_view.h"

namespace tachyon::halo2_api::bn254 {

RationalEvals::RationalEvals()
    : evals_(tachyon_bn254_univariate_rational_evaluations_create()) {}

RationalEvals::~RationalEvals() {
  tachyon_bn254_univariate_rational_evaluations_destroy(evals_);
}

size_t RationalEvals::len() const {
  return tachyon_bn254_univariate_rational_evaluations_len(evals_);
}

std::unique_ptr<RationalEvalsView> RationalEvals::create_view(size_t start,
                                                              size_t len) {
  return std::make_unique<RationalEvalsView>(evals_, start, len);
}

std::unique_ptr<RationalEvals> RationalEvals::clone() const {
  return std::make_unique<RationalEvals>(
      tachyon_bn254_univariate_rational_evaluations_clone(evals_));
}

}  // namespace tachyon::halo2_api::bn254

#include "vendors/scroll_halo2/include/bn254_rational_evals_view.h"

#include "tachyon/base/logging.h"
#include "vendors/scroll_halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

RationalEvalsView::RationalEvalsView(
    tachyon_bn254_univariate_rational_evaluations* evals, size_t start,
    size_t len)
    : evals_(evals), start_(start), len_(len) {
  CHECK_GT(len, size_t{0});
}

void RationalEvalsView::set_zero(size_t idx) {
  CHECK_LT(start_ + idx, len_);
  tachyon_bn254_univariate_rational_evaluations_set_zero(evals_, start_ + idx);
}

void RationalEvalsView::set_trivial(size_t idx, const Fr& numerator) {
  CHECK_LT(start_ + idx, len_);
  tachyon_bn254_univariate_rational_evaluations_set_trivial(
      evals_, start_ + idx,
      reinterpret_cast<const tachyon_bn254_fr*>(&numerator));
}

void RationalEvalsView::set_rational(size_t idx, const Fr& numerator,
                                     const Fr& denominator) {
  CHECK_LT(start_ + idx, len_);
  tachyon_bn254_univariate_rational_evaluations_set_rational(
      evals_, start_ + idx,
      reinterpret_cast<const tachyon_bn254_fr*>(&numerator),
      reinterpret_cast<const tachyon_bn254_fr*>(&denominator));
}

void RationalEvalsView::evaluate(size_t idx, Fr& value) const {
  CHECK_LT(start_ + idx, len_);
  tachyon_bn254_univariate_rational_evaluations_evaluate(
      evals_, start_ + idx, reinterpret_cast<tachyon_bn254_fr*>(&value));
}

}  // namespace tachyon::halo2_api::bn254

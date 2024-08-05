#include "vendors/scroll_halo2/include/bn254_evals.h"

#include "vendors/scroll_halo2/src/bn254.rs.h"

namespace tachyon::halo2_api::bn254 {

Evals::Evals() : evals_(tachyon_bn254_univariate_evaluations_create()) {}

Evals::~Evals() { tachyon_bn254_univariate_evaluations_destroy(evals_); }

size_t Evals::len() const {
  return tachyon_bn254_univariate_evaluations_len(evals_);
}

void Evals::set_value(size_t idx, const Fr& fr) {
  tachyon_bn254_univariate_evaluations_set_value(
      evals_, idx, reinterpret_cast<const tachyon_bn254_fr*>(&fr));
}

std::unique_ptr<Evals> Evals::clone() const {
  return std::make_unique<Evals>(
      tachyon_bn254_univariate_evaluations_clone(evals_));
}

std::unique_ptr<Evals> zero_evals() { return std::make_unique<Evals>(); }

}  // namespace tachyon::halo2_api::bn254

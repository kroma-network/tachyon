#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::math::internal {

class MSMInputProvider {
 public:
  void set_needs_align(bool needs_align) { needs_align_ = needs_align; }

  absl::Span<const bn254::G1AffinePoint> bases() const { return bases_; }
  absl::Span<const bn254::Fr> scalars() const { return scalars_; }

  void Inject(const tachyon_bn254_g1_point2* bases_in, size_t bases_len,
              const tachyon_bn254_fr* scalars_in, size_t scalars_len);

  void Inject(const tachyon_bn254_g1_affine* bases_in, size_t bases_len,
              const tachyon_bn254_fr* scalars_in, size_t scalars_len);

 private:
  bool needs_align_ = false;
  absl::Span<const bn254::G1AffinePoint> bases_;
  absl::Span<const bn254::Fr> scalars_;
  std::vector<bn254::G1AffinePoint> bases_owned_;
  std::vector<bn254::Fr> scalars_owned_;
};

}  // namespace tachyon::math::internal

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_MSM_MSM_INPUT_PROVIDER_H_

#include "tachyon/c/math/elliptic_curves/msm/msm_input_provider.h"

#include "absl/numeric/bits.h"

namespace tachyon::math::internal {

void MSMInputProvider::Inject(const tachyon_bn254_g1_point2* bases_in,
                              size_t bases_len,
                              const tachyon_bn254_fr* scalars_in,
                              size_t scalars_len) {
  absl::Span<const Point2<bn254::Fq>> points(
      reinterpret_cast<const Point2<bn254::Fq>*>(bases_in), bases_len);
  size_t size = 0;
  if (needs_align_) {
    size = absl::bit_ceil(bases_len);
    bases_owned_.resize(size);
    for (size_t i = bases_len; i < size; ++i) {
      bases_owned_[i] = bn254::G1AffinePoint::Zero();
    }
  } else {
    bases_owned_.resize(bases_len);
  }
  for (size_t i = 0; i < bases_len; ++i) {
    bases_owned_[i] = bn254::G1AffinePoint(
        points[i], points[i].x.IsZero() && points[i].y.IsZero());
  }
  bases_ = absl::MakeConstSpan(bases_owned_);

  if (needs_align_) {
    scalars_owned_.resize(size);
    for (size_t i = 0; i < scalars_len; ++i) {
      scalars_owned_[i] = reinterpret_cast<const bn254::Fr*>(scalars_in)[i];
    }
    for (size_t i = scalars_len; i < size; ++i) {
      scalars_owned_[i] = bn254::Fr::Zero();
    }
    scalars_ = absl::MakeConstSpan(scalars_owned_);
  } else {
    scalars_ = absl::MakeConstSpan(
        reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  }
}

void MSMInputProvider::Inject(const tachyon_bn254_g1_affine* bases_in,
                              size_t bases_len,
                              const tachyon_bn254_fr* scalars_in,
                              size_t scalars_len) {
  if (needs_align_) {
    size_t size = absl::bit_ceil(bases_len);
    bases_owned_.resize(size);
    for (size_t i = 0; i < bases_len; ++i) {
      bases_owned_[i] =
          reinterpret_cast<const bn254::G1AffinePoint*>(bases_in)[i];
    }
    for (size_t i = bases_len; i < size; ++i) {
      bases_owned_[i] = bn254::G1AffinePoint::Zero();
    }
    scalars_owned_.resize(size);
    for (size_t i = 0; i < scalars_len; ++i) {
      scalars_owned_[i] = reinterpret_cast<const bn254::Fr*>(scalars_in)[i];
    }
    for (size_t i = scalars_len; i < size; ++i) {
      scalars_owned_[i] = bn254::Fr::Zero();
    }
    bases_ = absl::MakeConstSpan(bases_owned_);
    scalars_ = absl::MakeConstSpan(scalars_owned_);
  } else {
    bases_ = absl::MakeConstSpan(
        reinterpret_cast<const bn254::G1AffinePoint*>(bases_in), bases_len);
    scalars_ = absl::MakeConstSpan(
        reinterpret_cast<const bn254::Fr*>(scalars_in), scalars_len);
  }
}

}  // namespace tachyon::math::internal

#ifndef TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_TRAITS_H_
#define TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_TRAITS_H_

#include <type_traits>

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::crypto {

template <typename T, typename SFINAE = void>
struct TranscriptTraits;

template <typename Curve>
struct TranscriptTraits<math::AffinePoint<Curve>> {
  constexpr static bool kFieldAndCommitmentAreSameType = false;

  using Field = typename math::AffinePoint<Curve>::ScalarField;
};

template <typename F>
struct TranscriptTraits<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>> {
  constexpr static bool kFieldAndCommitmentAreSameType = true;

  using Field = F;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_TRANSCRIPTS_TRANSCRIPT_TRAITS_H_

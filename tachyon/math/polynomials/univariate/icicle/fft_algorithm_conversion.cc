#include "tachyon/math/polynomials/univariate/icicle/fft_algorithm_conversion.h"

#include "tachyon/base/logging.h"

namespace tachyon::math {

::ntt::NttAlgorithm FFTAlgorithmToIcicleNTTAlgorithm(FFTAlgorithm algorithm) {
  switch (algorithm) {
    case FFTAlgorithm::kRadix2:
      return ::ntt::NttAlgorithm::Radix2;
    case FFTAlgorithm::kMixedRadix:
      return ::ntt::NttAlgorithm::MixedRadix;
  }
  NOTREACHED();
  return ::ntt::NttAlgorithm::Auto;
}

}  // namespace tachyon::math

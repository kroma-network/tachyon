#include "tachyon/math/polynomials/univariate/fft_algorithm.h"

#include "tachyon/base/logging.h"

namespace tachyon::math {

std::string FFTAlgorithmToString(FFTAlgorithm algorithm) {
  switch (algorithm) {
    case FFTAlgorithm::kRadix2:
      return "Radix2";
    case FFTAlgorithm::kMixedRadix:
      return "MixedRadix";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::math

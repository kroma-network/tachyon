#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_FFT_ALGORITHM_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_FFT_ALGORITHM_H_

#include <string>

#include "tachyon/export.h"

namespace tachyon::math {

enum class FFTAlgorithm {
  kRadix2,
  kMixedRadix,
};

TACHYON_EXPORT std::string FFTAlgorithmToString(FFTAlgorithm algorithm);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_FFT_ALGORITHM_H_

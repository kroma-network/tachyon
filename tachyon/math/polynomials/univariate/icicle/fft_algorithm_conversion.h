#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_FFT_ALGORITHM_CONVERSION_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_FFT_ALGORITHM_CONVERSION_H_

#include "third_party/icicle/include/ntt/ntt_algorithm.h"

#include "tachyon/export.h"
#include "tachyon/math/polynomials/univariate/fft_algorithm.h"

namespace tachyon::math {

TACHYON_EXPORT ::ntt::NttAlgorithm FFTAlgorithmToIcicleNTTAlgorithm(
    FFTAlgorithm algorithm);

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_FFT_ALGORITHM_CONVERSION_H_

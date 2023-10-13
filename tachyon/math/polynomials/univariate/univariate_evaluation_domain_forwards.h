// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FORWARDS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FORWARDS_H_

#include <stddef.h>

namespace tachyon::math {

template <typename F, size_t MaxDegree>
class UnivariateEvaluationDomain;

template <typename F, size_t MaxDegree>
class Radix2EvaluationDomain;

template <typename F, size_t MaxDegree>
class MixedRadixEvaluationDomain;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FORWARDS_H_

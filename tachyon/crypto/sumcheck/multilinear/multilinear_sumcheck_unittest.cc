// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/crypto/sumcheck/multilinear/multilinear_sumcheck.h"

#include "gtest/gtest.h"

#include "tachyon/base/range.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/linear_combination.h"
#include "tachyon/math/polynomials/multivariate/multilinear_dense_evaluations.h"

namespace tachyon::crypto {
namespace {

using F = math::GF7;

class MultilinearSumcheckTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(MultilinearSumcheckTest, TestInteractiveProtocolSmall) {
  // |SumcheckVerifier| runs in-depth logic in |InterpolateUniPoly()|
  constexpr size_t kNumVariables = 2;
  const base::Range<size_t> kMaxEvaluationsInTermRange(2, 3);
  constexpr size_t kMaxPossibleEvaluations = 3;
  size_t kNumTerms = 2;
  using MLE = math::MultilinearDenseEvaluations<F, kNumVariables>;
  const math::LinearCombination<MLE> linear_combination =
      math::LinearCombination<MLE>::Random(
          kNumVariables, kMaxEvaluationsInTermRange, kNumTerms);

  EXPECT_TRUE(
      MultilinearSumcheck<MLE>::RunInteractiveProtocol<kMaxPossibleEvaluations>(
          linear_combination, kNumVariables));
}

TEST_F(MultilinearSumcheckTest, TestInteractiveProtocolBig) {
  // |SumcheckVerifier| returns early for |InterpolateUniPoly()|
  constexpr size_t kNumVariables = 10;
  const base::Range<size_t> kMaxEvaluationsInTermRange(6, 30);
  constexpr size_t kMaxPossibleEvaluations = 30;
  size_t kNumTerms = 20;
  using MLE = math::MultilinearDenseEvaluations<F, kNumVariables>;
  const math::LinearCombination<MLE> linear_combination =
      math::LinearCombination<MLE>::Random(
          kNumVariables, kMaxEvaluationsInTermRange, kNumTerms);

  EXPECT_TRUE(
      MultilinearSumcheck<MLE>::RunInteractiveProtocol<kMaxPossibleEvaluations>(
          linear_combination, kNumVariables));
}

}  // namespace tachyon::crypto

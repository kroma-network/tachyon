// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_TEST_EVALUATOR_TEST_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_TEST_EVALUATOR_TEST_H_

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::zk {

constexpr size_t kMaxSize = 6;

using GF7 = math::GF7;
using Poly = math::UnivariateDensePolynomial<GF7, kMaxSize>;
using Coeffs = math::UnivariateDenseCoefficients<GF7, kMaxSize>;

class EvaluatorTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::GF7::Init(); }

  void SetUp() override {
    fixed_polys_.push_back(Poly(Coeffs({GF7(3)})));
    fixed_polys_.push_back(Poly(Coeffs({GF7(2), GF7(4)})));

    advice_polys_.push_back(Poly(Coeffs({GF7(1)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(2), GF7(3)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(4), GF7(5)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(6), GF7(1), GF7(2)})));

    instance_polys_.push_back(Poly(Coeffs({GF7(1)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(4), GF7(1)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(2), GF7(3)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(5), GF7(6)})));

    challenges_.push_back(GF7(1));
    challenges_.push_back(GF7(3));
    challenges_.push_back(GF7(4));
    challenges_.push_back(GF7(5));
  }

 protected:
  std::vector<Poly> fixed_polys_;
  std::vector<Poly> advice_polys_;
  std::vector<Poly> instance_polys_;
  std::vector<GF7> challenges_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_TEST_EVALUATOR_TEST_H_

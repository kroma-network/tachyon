#include "tachyon/math/polynomials/multivariate/linear_combination.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon::math {
namespace {

const size_t kMaxDegree = 4;

using F = GF7;
using Poly = MultilinearExtension<MultilinearDenseEvaluations<F, kMaxDegree>>;
using Point = Poly::Point;
using PolyPtr = std::shared_ptr<Poly>;

class LinearCombinationTest : public FiniteFieldTest<F> {};

}  // namespace

// Tests |AddTerm()| and |SumOfProducts()|
TEST_F(LinearCombinationTest, AddTermAndSumOfProducts) {
  LinearCombination<Poly> linear_combination(kMaxDegree);
  F coefficient1 = F::Random();

  // |evals| takes in 5 random polys and an additional 5 of the same random
  // poly |test_multiple| to test logic of inputting the same random poly
  PolyPtr test_multiple = std::make_shared<Poly>(Poly::Random(kMaxDegree));
  std::vector<PolyPtr> evals1 =
      base::CreateVector(10, [test_multiple](size_t i) {
        if (i % 2 != 0) return std::make_shared<Poly>(Poly::Random(kMaxDegree));
        return test_multiple;
      });
  linear_combination.AddTerm(coefficient1, evals1);

  Point evaluation_point = base::CreateVector(kMaxDegree, F::Random());

  // Set up the expected evaluation result for this 1 term
  F expected_evaluation =
      coefficient1 *
      std::accumulate(evals1.begin(), evals1.end(), F::One(),
                      [&evaluation_point](F& acc, const PolyPtr eval) {
                        return acc *= eval->Evaluate(evaluation_point);
                      });

  // Test validity of |AddTerm()| and |SumOfProducts()| on 1 term
  EXPECT_EQ(linear_combination.SumOfProducts(evaluation_point),
            expected_evaluation);

  // Test |AddTerm()| and |SumOfProducts| on 2 terms
  F coefficient2 = F::Random();
  std::vector<PolyPtr> evals2 =
      base::CreateVector(5, std::make_shared<Poly>(Poly::Random(kMaxDegree)));
  linear_combination.AddTerm(coefficient2, evals2);
  expected_evaluation +=
      coefficient2 *
      std::accumulate(evals2.begin(), evals2.end(), F::One(),
                      [&evaluation_point](F& acc, const PolyPtr eval) {
                        return acc *= eval->Evaluate(evaluation_point);
                      });
  EXPECT_EQ(linear_combination.SumOfProducts(evaluation_point),
            expected_evaluation);
}

}  // namespace tachyon::math

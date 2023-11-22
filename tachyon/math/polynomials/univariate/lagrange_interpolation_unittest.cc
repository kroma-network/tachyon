#include "tachyon/math/polynomials/univariate/lagrange_interpolation.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

TEST(LagrangeInterpolationTest, LagrangeInterpolate) {
  using F = math::GF7;
  F::Init();

  // Lagrange polynomial for (1, 2), (2, 0), (3, 0)
  std::vector<F> points_3({F(1), F(2), F(3)});
  std::vector<F> evals_3({F(2), F(0), F(0)});

  // Expected: (x - 2)(x - 3) = x² - 5x + 6 = x² + 2x + 6
  UnivariateDensePolynomial<F, 2> expected_2d_poly =
      UnivariateDensePolynomial<F, 2>(
          UnivariateDenseCoefficients<F, 2>({F(6), F(2), F(1)}));

  UnivariateDensePolynomial<F, 2> actual_2d_poly;
  EXPECT_TRUE(LagrangeInterpolate(points_3, evals_3, &actual_2d_poly));

  EXPECT_EQ(expected_2d_poly, actual_2d_poly);

  // Lagrange polynomial for (0,4), (1,3), (2,5), (6,2)
  std::vector<F> points_4({F(0), F(1), F(2), F(6)});
  std::vector<F> evals_4({F(4), F(3), F(5), F(2)});

  // Expected: x³ + 2x² + 3x + 4
  UnivariateDensePolynomial<F, 3> expected_3d_poly =
      UnivariateDensePolynomial<F, 3>(
          UnivariateDenseCoefficients<F, 3>({F(4), F(3), F(2), F(1)}));

  UnivariateDensePolynomial<F, 3> actual_3d_poly;
  EXPECT_TRUE(LagrangeInterpolate(points_4, evals_4, &actual_3d_poly));

  EXPECT_EQ(expected_3d_poly, actual_3d_poly);
}

}  // namespace tachyon::math

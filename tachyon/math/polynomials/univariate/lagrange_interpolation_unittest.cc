#include "tachyon/math/polynomials/univariate/lagrange_interpolation.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::math {

TEST(LagrangeInterpolationTest, LagrangeInterpolate) {
  using F = bn254::Fr;
  F::Init();

#if defined(TACHYON_HAS_OPENMP)
  constexpr size_t kDegree = 128;
#else
  constexpr size_t kDegree = 4;
#endif

  std::vector<F> points =
      base::CreateVector(kDegree + 1, [](size_t i) { return F(i); });
  std::vector<F> evals =
      base::CreateVector(kDegree + 1, []() { return F::Random(); });

  UnivariateDensePolynomial<F, kDegree> poly;
  EXPECT_TRUE(LagrangeInterpolate(points, evals, &poly));

  OMP_PARALLEL_FOR(size_t i = 0; i < points.size(); ++i) {
    EXPECT_EQ(poly.Evaluate(points[i]), evals[i]);
  }
}

}  // namespace tachyon::math

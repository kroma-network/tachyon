#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateDensePolynomialTest : public testing::Test {
 public:
  using Poly = UnivariateDensePolynomial<bn254::Fr, c::math::kMaxDegree>;

  static void SetUpTestSuite() { bn254::Fr::Init(); }

  void SetUp() override {
    Poly* cpp_poly = new Poly(Poly::Random(kDegree));
    poly_ =
        reinterpret_cast<tachyon_bn254_univariate_dense_polynomial*>(cpp_poly);
  }

  void TearDown() override {
    tachyon_bn254_univariate_dense_polynomial_destroy(poly_);
  }

 protected:
  tachyon_bn254_univariate_dense_polynomial* poly_;
};

}  // namespace

TEST_F(UnivariateDensePolynomialTest, Clone) {
  if (reinterpret_cast<Poly&>(*poly_)[0] != nullptr) {
    tachyon_bn254_univariate_dense_polynomial* poly_clone =
        tachyon_bn254_univariate_dense_polynomial_clone(poly_);
    *reinterpret_cast<Poly&>(*poly_)[0] += bn254::Fr::One();
    EXPECT_NE((reinterpret_cast<Poly&>(*poly_))[0],
              (reinterpret_cast<Poly&>(*poly_clone))[0]);
    tachyon_bn254_univariate_dense_polynomial_destroy(poly_clone);
  } else {
    GTEST_SKIP() << "This test assumes that the coefficient for the 0th degree "
                    "is not zero";
  }
}

}  // namespace tachyon::math

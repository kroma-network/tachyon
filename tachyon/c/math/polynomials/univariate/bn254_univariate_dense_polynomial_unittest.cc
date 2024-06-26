#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/polynomials/univariate/bn254_univariate_dense_polynomial_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateDensePolynomialTest : public FiniteFieldTest<bn254::Fr> {
 public:
  using Poly = UnivariateDensePolynomial<bn254::Fr, c::math::kMaxDegree>;

  void SetUp() override {
    Poly* cpp_poly = new Poly(Poly::Random(kDegree));
    poly_ = c::base::c_cast(cpp_poly);
  }

  void TearDown() override {
    tachyon_bn254_univariate_dense_polynomial_destroy(poly_);
  }

 protected:
  tachyon_bn254_univariate_dense_polynomial* poly_;
};

}  // namespace

TEST_F(UnivariateDensePolynomialTest, Clone) {
  if (c::base::native_cast(*poly_).NumElements() > 0) {
    tachyon_bn254_univariate_dense_polynomial* poly_clone =
        tachyon_bn254_univariate_dense_polynomial_clone(poly_);
    // NOTE(chokobole): It's safe to access since we checked |NumElements()| is
    // greater than 0.
    c::base::native_cast(*poly_).at(0) += bn254::Fr::One();
    EXPECT_NE((c::base::native_cast(*poly_))[0],
              (c::base::native_cast(*poly_clone))[0]);
    tachyon_bn254_univariate_dense_polynomial_destroy(poly_clone);
  } else {
    GTEST_SKIP() << "This test assumes that the coefficient for the 0th degree "
                    "is not empty";
  }
}

}  // namespace tachyon::math

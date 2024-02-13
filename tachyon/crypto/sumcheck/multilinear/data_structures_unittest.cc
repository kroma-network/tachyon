#include "data_structures.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/multivariate/multilinear_dense_evaluations.h"
#include "tachyon/math/polynomials/multivariate/multilinear_extension.h"

namespace tachyon {

namespace crypto::sumcheck {

namespace {

const size_t kMaxDegree = 4;

using Poly = math::MultilinearExtension<
    math::MultilinearDenseEvaluations<math::GF7, kMaxDegree>>;
using Evals = math::MultilinearDenseEvaluations<math::GF7, kMaxDegree>;
using ProductList = ListOfProductsOfPolynomials<Poly>;
using Point = math::GF7;

}  // namespace

// Tests constructors & `Info()`
TEST(ListOfProductsOfPolynomials, Constructors) {
  // Test default constructor
  ProductList a;
  PolynomialInfo expected_a{0, 0};
  EXPECT_EQ(a.Info(), expected_a);
  // Test constructor passing `num_variables`
  ProductList b(kMaxDegree);
  PolynomialInfo expected_b{0, kMaxDegree};
  EXPECT_EQ(b.Info(), expected_b);
  // Test copy constructor
  ProductList c = b;
  EXPECT_EQ(c.Info(), expected_b);
}

// Tests `AddTerm()` and `Evaluate()`
TEST(ListOfProductsOfPolynomials, AddTermAndEvaluate) {
  ProductList product_list(kMaxDegree);
  Point coefficient_1 = Point::Random();

  // `evals` takes in 5 Random Polys and an additional 5 of the same Random
  // Poly `test_multiple` to test logic of inputting the same Random Poly
  std::vector<std::shared_ptr<Poly>> evals_1;
  evals_1.reserve(10);
  std::shared_ptr<Poly> test_multiple =
      std::make_shared<Poly>(Poly::Random(kMaxDegree));
  for (size_t i = 0; i < 5; ++i) {
    evals_1.push_back(std::make_shared<Poly>(Poly::Random(kMaxDegree)));
    evals_1.push_back(test_multiple);
  }
  // Term added
  product_list.AddTerm(coefficient_1, evals_1);

  // Create an evaluation point
  std::vector<Point> evaluation_point;
  evaluation_point.reserve(kMaxDegree);
  for (size_t i = 0; i < kMaxDegree; ++i) {
    evaluation_point.push_back(Point::Random());
  }

  // Set up the expected evaluation result for this 1 term
  Point multiplied_evals = Point::One();
  for (std::shared_ptr<Poly> eval : evals_1) {
    multiplied_evals *= eval->Evaluate(evaluation_point);
  }
  Point expected_evaluation = coefficient_1 * multiplied_evals;

  // Test validity of `AddTerm()` and `Evaluate()` on 1 term
  CHECK_EQ(product_list.Evaluate(evaluation_point), expected_evaluation);

  // Test `AddTerm()` and `Evaluate` on 2 Terms:
  Point coefficient_2 = Point::Random();
  std::vector<std::shared_ptr<Poly>> evals_2;
  evals_2.reserve(5);
  for (size_t i = 0; i < 5; ++i) {
    evals_2.push_back(std::make_shared<Poly>(Poly::Random(kMaxDegree)));
  }
  product_list.AddTerm(coefficient_2, evals_2);
  multiplied_evals = Point::One();
  for (std::shared_ptr<Poly> eval : evals_2) {
    multiplied_evals *= eval->Evaluate(evaluation_point);
  }
  expected_evaluation += coefficient_2 * multiplied_evals;
  CHECK_EQ(product_list.Evaluate(evaluation_point), expected_evaluation);

  // Safety check for `max_multiplicands`
  PolynomialInfo expected_product_list{10, kMaxDegree};
  EXPECT_EQ(product_list.Info(), expected_product_list);
}

}  // namespace crypto::sumcheck

namespace base {

TEST(PolynomialInfo, Copyable) {
  // TODO : implement Random()
  crypto::sumcheck::PolynomialInfo expected = {10, 4};

  std::vector<uint8_t> vec;
  vec.resize(base::EstimateSize(expected));
  base::Buffer write_buf(vec.data(), vec.size());
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  crypto::sumcheck::PolynomialInfo value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

}  // namespace base
}  // namespace tachyon

#include "gtest/gtest.h"

#include "tachyon/base/random.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

using F = math::GF7;

namespace {

class ExpressionTest : public math::FiniteFieldTest<F> {
 public:
  void SetUp() override {
    expressions_.push_back(ExpressionFactory<F>::Constant(F(1)));
    expressions_.push_back(
        ExpressionFactory<F>::Selector(plonk::Selector::Simple(1)));
    expressions_.push_back(
        ExpressionFactory<F>::Selector(plonk::Selector::Complex(2)));
    expressions_.push_back(ExpressionFactory<F>::Fixed(
        plonk::FixedQuery(1, Rotation(1), plonk::FixedColumnKey(1))));
    expressions_.push_back(ExpressionFactory<F>::Advice(plonk::AdviceQuery(
        1, Rotation(1), plonk::AdviceColumnKey(1, plonk::Phase(0)))));
    expressions_.push_back(ExpressionFactory<F>::Instance(
        plonk::InstanceQuery(1, Rotation(1), plonk::InstanceColumnKey(1))));
    expressions_.push_back(
        ExpressionFactory<F>::Challenge(plonk::Challenge(1, plonk::Phase(0))));
    expressions_.push_back(
        ExpressionFactory<F>::Negated(ExpressionFactory<F>::Constant(F(1))));
    expressions_.push_back(
        ExpressionFactory<F>::Sum(ExpressionFactory<F>::Constant(F(1)),
                                  ExpressionFactory<F>::Constant(F(2))));
    expressions_.push_back(
        ExpressionFactory<F>::Product(ExpressionFactory<F>::Constant(F(3)),
                                      ExpressionFactory<F>::Constant(F(4))));
    expressions_.push_back(ExpressionFactory<F>::Scaled(
        ExpressionFactory<F>::Constant(F(5)), F(6)));
  }

 protected:
  std::vector<std::unique_ptr<Expression<F>>> expressions_;
};

}  // namespace

TEST_F(ExpressionTest, ArithmeticOperatorWithClone) {
  std::unique_ptr<Expression<F>> left =
      base::UniformElement(expressions_)->Clone();
  std::unique_ptr<Expression<F>> right =
      base::UniformElement(expressions_)->Clone();

  if (left->ContainsSimpleSelector() && right->ContainsSimpleSelector()) {
    EXPECT_DEATH(left + right, "");
    EXPECT_DEATH(left - right, "");
    EXPECT_DEATH(left * right, "");
  } else {
    std::unique_ptr<Expression<F>> add = left + right;
    EXPECT_EQ(*add->ToSum()->left(), *left);
    EXPECT_EQ(*add->ToSum()->right(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);

    std::unique_ptr<Expression<F>> sub = left - right;
    EXPECT_EQ(*sub->ToSum()->left(), *left);
    EXPECT_EQ(*sub->ToSum()->right()->ToNegated()->expr(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);

    std::unique_ptr<Expression<F>> mul = left * right;
    EXPECT_EQ(*mul->ToProduct()->left(), *left);
    EXPECT_EQ(*mul->ToProduct()->right(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);
  }

  F scale = F::Random();
  std::unique_ptr<Expression<F>> scaled = left * scale;
  EXPECT_EQ(*scaled->ToScaled()->expr(), *left);
  EXPECT_EQ(scaled->ToScaled()->scale(), scale);
  EXPECT_TRUE(left);

  std::unique_ptr<Expression<F>> neg = -left;
  EXPECT_EQ(*neg->ToNegated()->expr(), *left);
  EXPECT_TRUE(left);
}

TEST_F(ExpressionTest, ArithmeticOperatorWithMove) {
  std::unique_ptr<Expression<F>> left =
      base::UniformElement(expressions_)->Clone();
  std::unique_ptr<Expression<F>> right =
      base::UniformElement(expressions_)->Clone();

  if (left->ContainsSimpleSelector() && right->ContainsSimpleSelector()) {
    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) + std::move(right_tmp), "");
    }
    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) - std::move(right_tmp), "");
    }
    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) * std::move(right_tmp), "");
    }
  } else {
    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      std::unique_ptr<Expression<F>> add =
          std::move(left_tmp) + std::move(right_tmp);
      EXPECT_EQ(*add->ToSum()->left(), *left);
      EXPECT_EQ(*add->ToSum()->right(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }

    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      std::unique_ptr<Expression<F>> sub =
          std::move(left_tmp) - std::move(right_tmp);
      EXPECT_EQ(*sub->ToSum()->left(), *left);
      EXPECT_EQ(*sub->ToSum()->right()->ToNegated()->expr(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }

    {
      std::unique_ptr<Expression<F>> left_tmp = left->Clone();
      std::unique_ptr<Expression<F>> right_tmp = right->Clone();
      std::unique_ptr<Expression<F>> mul =
          std::move(left_tmp) * std::move(right_tmp);
      EXPECT_EQ(*mul->ToProduct()->left(), *left);
      EXPECT_EQ(*mul->ToProduct()->right(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }
  }

  {
    std::unique_ptr<Expression<F>> left_tmp = left->Clone();
    F scale = F::Random();
    std::unique_ptr<Expression<F>> scaled = std::move(left_tmp) * scale;
    EXPECT_EQ(*scaled->ToScaled()->expr(), *left);
    EXPECT_EQ(scaled->ToScaled()->scale(), scale);
    EXPECT_FALSE(left_tmp);
  }

  {
    std::unique_ptr<Expression<F>> left_tmp = left->Clone();
    std::unique_ptr<Expression<F>> neg = -std::move(left_tmp);
    EXPECT_EQ(*neg->ToNegated()->expr(), *left);
    EXPECT_FALSE(left_tmp);
  }
}

}  // namespace tachyon::zk

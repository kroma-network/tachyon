#include "gtest/gtest.h"

#include "tachyon/base/random.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

namespace {

class ExpressionTest : public math::FiniteFieldTest<Fr> {
 public:
  void SetUp() override {
    expressions_.push_back(ExpressionFactory<Fr>::Constant(Fr(1)));
    expressions_.push_back(
        ExpressionFactory<Fr>::Selector(plonk::Selector::Simple(1)));
    expressions_.push_back(
        ExpressionFactory<Fr>::Selector(plonk::Selector::Complex(2)));
    expressions_.push_back(ExpressionFactory<Fr>::Fixed(
        plonk::FixedQuery(1, Rotation(1), plonk::FixedColumnKey(1))));
    expressions_.push_back(ExpressionFactory<Fr>::Advice(plonk::AdviceQuery(
        1, Rotation(1), plonk::AdviceColumnKey(1, plonk::Phase(0)))));
    expressions_.push_back(ExpressionFactory<Fr>::Instance(
        plonk::InstanceQuery(1, Rotation(1), plonk::InstanceColumnKey(1))));
    expressions_.push_back(
        ExpressionFactory<Fr>::Challenge(plonk::Challenge(1, plonk::Phase(0))));
    expressions_.push_back(
        ExpressionFactory<Fr>::Negated(ExpressionFactory<Fr>::Constant(Fr(1))));
    expressions_.push_back(
        ExpressionFactory<Fr>::Sum(ExpressionFactory<Fr>::Constant(Fr(1)),
                                   ExpressionFactory<Fr>::Constant(Fr(2))));
    expressions_.push_back(
        ExpressionFactory<Fr>::Product(ExpressionFactory<Fr>::Constant(Fr(3)),
                                       ExpressionFactory<Fr>::Constant(Fr(4))));
    expressions_.push_back(ExpressionFactory<Fr>::Scaled(
        ExpressionFactory<Fr>::Constant(Fr(5)), Fr(6)));
  }

 protected:
  std::vector<std::unique_ptr<Expression<Fr>>> expressions_;
};

}  // namespace

TEST_F(ExpressionTest, ArithmeticOperatorWithClone) {
  std::unique_ptr<Expression<Fr>> left =
      base::UniformElement(expressions_)->Clone();
  std::unique_ptr<Expression<Fr>> right =
      base::UniformElement(expressions_)->Clone();

  if (left->ContainsSimpleSelector() && right->ContainsSimpleSelector()) {
    EXPECT_DEATH(left + right, "");
    EXPECT_DEATH(left - right, "");
    EXPECT_DEATH(left * right, "");
  } else {
    std::unique_ptr<Expression<Fr>> add = left + right;
    EXPECT_EQ(*add->ToSum()->left(), *left);
    EXPECT_EQ(*add->ToSum()->right(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);

    std::unique_ptr<Expression<Fr>> sub = left - right;
    EXPECT_EQ(*sub->ToSum()->left(), *left);
    EXPECT_EQ(*sub->ToSum()->right()->ToNegated()->expr(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);

    std::unique_ptr<Expression<Fr>> mul = left * right;
    EXPECT_EQ(*mul->ToProduct()->left(), *left);
    EXPECT_EQ(*mul->ToProduct()->right(), *right);
    EXPECT_TRUE(left);
    EXPECT_TRUE(right);
  }

  Fr scale = Fr::Random();
  std::unique_ptr<Expression<Fr>> scaled = left * scale;
  EXPECT_EQ(*scaled->ToScaled()->expr(), *left);
  EXPECT_EQ(scaled->ToScaled()->scale(), scale);
  EXPECT_TRUE(left);

  std::unique_ptr<Expression<Fr>> neg = -left;
  EXPECT_EQ(*neg->ToNegated()->expr(), *left);
  EXPECT_TRUE(left);
}

TEST_F(ExpressionTest, ArithmeticOperatorWithMove) {
  std::unique_ptr<Expression<Fr>> left =
      base::UniformElement(expressions_)->Clone();
  std::unique_ptr<Expression<Fr>> right =
      base::UniformElement(expressions_)->Clone();

  if (left->ContainsSimpleSelector() && right->ContainsSimpleSelector()) {
    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) + std::move(right_tmp), "");
    }
    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) - std::move(right_tmp), "");
    }
    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      EXPECT_DEATH(std::move(left_tmp) * std::move(right_tmp), "");
    }
  } else {
    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      std::unique_ptr<Expression<Fr>> add =
          std::move(left_tmp) + std::move(right_tmp);
      EXPECT_EQ(*add->ToSum()->left(), *left);
      EXPECT_EQ(*add->ToSum()->right(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }

    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      std::unique_ptr<Expression<Fr>> sub =
          std::move(left_tmp) - std::move(right_tmp);
      EXPECT_EQ(*sub->ToSum()->left(), *left);
      EXPECT_EQ(*sub->ToSum()->right()->ToNegated()->expr(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }

    {
      std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
      std::unique_ptr<Expression<Fr>> right_tmp = right->Clone();
      std::unique_ptr<Expression<Fr>> mul =
          std::move(left_tmp) * std::move(right_tmp);
      EXPECT_EQ(*mul->ToProduct()->left(), *left);
      EXPECT_EQ(*mul->ToProduct()->right(), *right);
      EXPECT_FALSE(left_tmp);
      EXPECT_FALSE(right_tmp);
    }
  }

  {
    std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
    Fr scale = Fr::Random();
    std::unique_ptr<Expression<Fr>> scaled = std::move(left_tmp) * scale;
    EXPECT_EQ(*scaled->ToScaled()->expr(), *left);
    EXPECT_EQ(scaled->ToScaled()->scale(), scale);
    EXPECT_FALSE(left_tmp);
  }

  {
    std::unique_ptr<Expression<Fr>> left_tmp = left->Clone();
    std::unique_ptr<Expression<Fr>> neg = -std::move(left_tmp);
    EXPECT_EQ(*neg->ToNegated()->expr(), *left);
    EXPECT_FALSE(left_tmp);
  }
}

}  // namespace tachyon::zk

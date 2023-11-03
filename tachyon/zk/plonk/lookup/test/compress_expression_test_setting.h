#ifndef TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_
#define TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/simple_evaluator.h"

namespace tachyon::zk {

class CompressExpressionTestSetting : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 5) - 1;
  constexpr static size_t kNumCoeffs = kMaxDegree + 1;

  using PCS =
      crypto::KZGCommitmentScheme<math::bls12_381::G1AffinePoint,
                                  math::bls12_381::G2AffinePoint, kMaxDegree>;

  using F = PCS::Field;
  using Poly = PCS::Poly;
  using Evals = PCS::Evals;
  using Domain = PCS::Domain;

  static void SetUpTestSuite() { math::bls12_381::G1Curve::Init(); }

  void SetUp() override {
    domain_ = math::UnivariateEvaluationDomainFactory<F, kMaxDegree>::Create(
        kNumCoeffs);

    SimpleEvaluator<Evals>::Arguments arguments(
        &advice_values_, &fixed_values_, &instance_values_, &challenges_);
    evaluator_ = {0, static_cast<int32_t>(domain_->size()), 1, arguments};
    theta_ = F(2);
  }

 protected:
  std::unique_ptr<Domain> domain_;
  SimpleEvaluator<Evals> evaluator_;
  std::vector<Evals> advice_values_;
  std::vector<Evals> fixed_values_;
  std::vector<Evals> instance_values_;
  std::vector<F> challenges_;
  F theta_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_TEST_COMPRESS_EXPRESSION_TEST_SETTING_H_

// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/lookup/permute_expression_pair.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g2.h"

namespace tachyon::zk {

constexpr size_t kMaxDegree = (size_t{1} << 5) - 1;
constexpr size_t kDomainSize = kMaxDegree + 1;
constexpr size_t kBlindingFactors = 5;
constexpr size_t kUsableRows = kDomainSize - (kBlindingFactors + 1);

using PCS =
    crypto::KZGCommitmentScheme<math::bls12_381::G1AffinePoint,
                                math::bls12_381::G2AffinePoint, kMaxDegree>;

using F = PCS::Field;
using Evals = PCS::Evals;

class PermuteExpressionPairTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bls12_381::G1Curve::Init(); }
};

TEST_F(PermuteExpressionPairTest, PermuteExpressionPairTest) {
  std::vector<F> table_evals =
      base::CreateVector(kDomainSize, []() { return F::Random(); });

  std::vector<F> input_evals =
      base::CreateVector(kDomainSize, [&table_evals]() {
        return table_evals[base::Uniform(
            base::Range<size_t>::Until(kUsableRows))];
      });

  EvalsPair<Evals> input(Evals(std::move(input_evals)),
                         Evals(std::move(table_evals)));
  EvalsPair<Evals> output;
  Error err =
      PermuteExpressionPair<PCS>(kDomainSize, kBlindingFactors, input, &output);
  ASSERT_EQ(err, Error::kNone);

  // sanity check brought from halo2
  std::optional<F> last;
  for (size_t i = 0; i < kUsableRows; ++i) {
    const F& perm_input_expr = *output.input()[i];
    const F& perm_table_coeff = *output.table()[i];

    if (perm_input_expr != perm_table_coeff) {
      EXPECT_EQ(perm_input_expr, last.value());
    }
    last = perm_input_expr;
  }
}

TEST_F(PermuteExpressionPairTest, PermuteExpressionPairTestWrong) {
  // set input_evals not included within table_evals;
  std::vector<F> input_evals =
      base::CreateVector(kDomainSize, [](size_t i) { return F(i * 2); });

  std::vector<F> table_evals =
      base::CreateVector(kDomainSize, [](size_t i) { return F(i * 3); });

  EvalsPair<Evals> input = {Evals(std::move(input_evals)),
                            Evals(std::move(table_evals))};
  EvalsPair<Evals> output;
  Error err =
      PermuteExpressionPair<PCS>(kDomainSize, kBlindingFactors, input, &output);
  ASSERT_EQ(err, Error::kConstraintSystemFailure);
}

}  // namespace tachyon::zk

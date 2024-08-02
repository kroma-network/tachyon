// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/lookup/halo2/permute_expression_pair.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/memory/reusing_allocator.h"
#include "tachyon/base/random.h"
#include "tachyon/zk/plonk/halo2/bn254_shplonk_prover_test.h"

namespace tachyon::zk::lookup::halo2 {

class PermuteExpressionPairTest : public plonk::halo2::BN254SHPlonkProverTest {
};

TEST_F(PermuteExpressionPairTest, PermuteExpressionPairTest) {
  prover_->blinder().set_blinding_factors(5);
  size_t n = prover_->pcs().N();
  RowIndex usable_rows = prover_->GetUsableRows();

  std::vector<F, base::memory::ReusingAllocator<F>> table_evals =
      base::CreatePmrVector(n, []() { return F::Random(); });

  std::vector<F, base::memory::ReusingAllocator<F>> input_evals =
      base::CreatePmrVector(n, [usable_rows, &table_evals]() {
        return table_evals[base::Uniform(
            base::Range<RowIndex>::Until(usable_rows))];
      });

  Pair<Evals> input(Evals(std::move(input_evals)),
                    Evals(std::move(table_evals)));
  Pair<Evals> output;
  ASSERT_TRUE(PermuteExpressionPair(prover_.get(), input, &output));

  // sanity check brought from halo2
  std::optional<F> last;
  for (size_t i = 0; i < usable_rows; ++i) {
    const F& perm_input_expr = output.input()[i];
    const F& perm_table_coeff = output.table()[i];

    if (perm_input_expr != perm_table_coeff) {
      EXPECT_EQ(perm_input_expr, last.value());
    }
    last = perm_input_expr;
  }
}

TEST_F(PermuteExpressionPairTest, PermuteExpressionPairTestWrong) {
  // set input_evals not included within table_evals;
  size_t n = prover_->pcs().N();
  std::vector<F, base::memory::ReusingAllocator<F>> input_evals =
      base::CreatePmrVector(n, [](size_t i) { return F(i * 2); });

  std::vector<F, base::memory::ReusingAllocator<F>> table_evals =
      base::CreatePmrVector(n, [](size_t i) { return F(i * 3); });

  Pair<Evals> input = {Evals(std::move(input_evals)),
                       Evals(std::move(table_evals))};
  Pair<Evals> output;
  ASSERT_FALSE(PermuteExpressionPair(prover_.get(), input, &output));
}

}  // namespace tachyon::zk::lookup::halo2

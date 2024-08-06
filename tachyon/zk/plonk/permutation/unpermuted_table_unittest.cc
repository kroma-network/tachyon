// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/unpermuted_table.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk::plonk {

namespace {

using F = math::bn254::Fr;

class UnpermutedTableTest : public math::FiniteFieldTest<F> {
 public:
  constexpr static RowIndex kRows = 32;
  constexpr static size_t kCols = 4;

  using Domain = math::UnivariateEvaluationDomain<F, kRows - 1>;
  using Evals = Domain::Evals;

  void SetUp() override {
    domain_ = Domain::Create(kRows);

    unpermuted_table_ =
        UnpermutedTable<Evals>::Construct(kCols, kRows, domain_.get());
  }

 protected:
  std::unique_ptr<Domain> domain_;
  UnpermutedTable<Evals> unpermuted_table_;
};

}  // namespace

TEST_F(UnpermutedTableTest, Construct) {
  const F& omega = domain_->group_gen();
  std::vector<F> omega_powers = domain_->GetRootsOfUnity(kRows, omega);

  const F delta = GetDelta<F>();
  for (size_t i = 1; i < kCols; ++i) {
    for (RowIndex j = 0; j < kRows; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], unpermuted_table_[Label(i, j)]);
    }
  }
}

TEST_F(UnpermutedTableTest, GetColumns) {
  const F& omega = domain_->group_gen();
  const F delta = GetDelta<F>();
  std::vector<F> omega_powers = domain_->GetRootsOfUnity(kRows, omega);
  for (F& omega_power : omega_powers) {
    omega_power *= delta;
  }

  base::Ref<const Evals> column = unpermuted_table_.GetColumn(1);
  for (RowIndex i = 0; i < kRows; ++i) {
    EXPECT_EQ(omega_powers[i], (*column)[i]);
  }

  std::vector<base::Ref<const Evals>> columns =
      unpermuted_table_.GetColumns(base::Range<size_t>(2, 4));
  for (size_t i = 0; i < 2; ++i) {
    for (RowIndex j = 0; j < kRows; ++j) {
      omega_powers[j] *= delta;
      EXPECT_EQ(omega_powers[j], (*columns[i])[j]);
    }
  }
}

}  // namespace tachyon::zk::plonk

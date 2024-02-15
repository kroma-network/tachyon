// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_utils.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk::plonk {

namespace {

template <typename PrimeField>
class PermutationUtilsTest : public math::FiniteFieldTest<PrimeField> {};

}  // namespace

using PrimeFieldTypes = testing::Types<math::bn254::Fq, math::bn254::Fr>;
TYPED_TEST_SUITE(PermutationUtilsTest, PrimeFieldTypes);

TYPED_TEST(PermutationUtilsTest, GetDelta) {
  using F = TypeParam;

  F delta = GetDelta<F>();
  EXPECT_NE(delta, F::One());
  EXPECT_EQ(delta.Pow(F::Config::kTrace), F::One());
}

}  // namespace tachyon::zk::plonk

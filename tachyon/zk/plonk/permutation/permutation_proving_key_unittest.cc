// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk::plonk {

namespace {

using F = math::bn254::Fr;

class PermutationProvingKeyTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(PermutationProvingKeyTest, Copyable) {
  constexpr static size_t N = 32;
  constexpr static size_t kMaxDegree = N - 1;

  using Domain = math::UnivariateEvaluationDomain<F, kMaxDegree>;
  using Poly = Domain::DensePoly;
  using Evals = Domain::Evals;
  using ProvingKey = PermutationProvingKey<Poly, Evals>;

  std::unique_ptr<Domain> domain = Domain::Create(N);

  ProvingKey expected(
      {domain->Random<Evals>(), domain->Random<Evals>(),
       domain->Random<Evals>()},
      {domain->Random<Poly>(), domain->Random<Poly>(), domain->Random<Poly>()});

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  ProvingKey value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::zk::plonk

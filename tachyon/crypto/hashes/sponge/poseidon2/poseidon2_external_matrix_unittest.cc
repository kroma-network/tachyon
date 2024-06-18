// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::crypto {

namespace {

template <typename Matrix>
class Poseidon2ExternalMatrixTest
    : public math::FiniteFieldTest<math::PackedMersenne31> {};

}  // namespace

using MatrixTypes =
    testing::Types<Poseidon2HorizenExternalMatrix<math::Mersenne31>,
                   Poseidon2Plonky3ExternalMatrix<math::Mersenne31>,
                   Poseidon2HorizenExternalMatrix<math::PackedMersenne31>,
                   Poseidon2Plonky3ExternalMatrix<math::PackedMersenne31>>;
TYPED_TEST_SUITE(Poseidon2ExternalMatrixTest, MatrixTypes);

TYPED_TEST(Poseidon2ExternalMatrixTest, DoApply) {
  using Matrix = TypeParam;
  using F = typename Matrix::Field;

  math::Vector<F> vector{{{F(1)}, {F(1)}, {F(1)}, {F(1)}}};
  math::Vector<F> vector2 = vector;
  Matrix::DoApply(vector2);
  EXPECT_EQ(Matrix::DoConstruct() * vector, vector2);
}

TYPED_TEST(Poseidon2ExternalMatrixTest, Apply) {
  using Matrix = TypeParam;
  using F = typename Matrix::Field;

  size_t sizes[] = {2, 3, 4, 8, 12, 16, 20, 24};

  for (size_t size : sizes) {
    math::Vector<F> vector(size);
    for (size_t i = 0; i < size; ++i) {
      vector(i, 0) = F::Random();
    }
    math::Vector<F> vector2 = vector;
    Poseidon2ExternalMatrix<Matrix>::Apply(vector2);
    EXPECT_EQ(Matrix::Construct(size) * vector, vector2);
  }

  size_t invalid_sizes[] = {0, 1, 5, 28};
  for (size_t size : invalid_sizes) {
    math::Vector<F> vector(size);
    EXPECT_DEATH(Poseidon2ExternalMatrix<Matrix>::Apply(vector), "");
    EXPECT_DEATH(Matrix::Construct(size), "");
  }
}

}  // namespace tachyon::crypto

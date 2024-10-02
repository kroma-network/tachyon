// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/matrix_utils.h"

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

  math::Vector<F, 4> vector{F(1), F(1), F(1), F(1)};
  std::array<F, 4> array = math::ToArray(vector);
  Poseidon2ExternalMatrix<Matrix>::Apply(array);
  EXPECT_EQ(math::ToArray(Matrix::template Construct<4>() * vector), array);
}

template <typename Matrix, size_t N>
void TestApply() {
  using F = typename Matrix::Field;

  math::Vector<F, N> vector = math::Vector<F, N>::Random();
  std::array<F, N> array = math::ToArray(vector);
  Poseidon2ExternalMatrix<Matrix>::Apply(array);
  EXPECT_EQ(math::ToArray(Matrix::template Construct<N>() * vector), array);
}

template <typename Matrix, size_t N>
void TestApplyDeath() {
  using F = typename Matrix::Field;

  std::array<F, N> array = base::CreateArray<N>([]() { return F::Random(); });
  EXPECT_DEATH(Poseidon2ExternalMatrix<Matrix>::Apply(array), "");
  EXPECT_DEATH(Matrix::template Construct<N>(), "");
}

TYPED_TEST(Poseidon2ExternalMatrixTest, Apply) {
  using Matrix = TypeParam;

  TestApply<Matrix, 2>();
  TestApply<Matrix, 3>();
  TestApply<Matrix, 4>();
  TestApply<Matrix, 8>();
  TestApply<Matrix, 12>();
  TestApply<Matrix, 16>();
  TestApply<Matrix, 20>();
  TestApply<Matrix, 24>();

  TestApplyDeath<Matrix, 0>();
  TestApplyDeath<Matrix, 1>();
  TestApplyDeath<Matrix, 5>();
  TestApplyDeath<Matrix, 28>();
}

}  // namespace tachyon::crypto

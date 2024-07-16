#include "tachyon/crypto/hashes/sponge/poseidon/sparse_mds_matrix.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

using F = math::bn254::Fr;

namespace {

class SparseMDSMatrixTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(SparseMDSMatrixTest, FromMDSMatrix) {
  math::Matrix<F> matrix = math::Matrix<F>::Random(5, 5);
  EXPECT_DEATH(SparseMDSMatrix<F>::FromMDSMatrix(matrix), "");
  matrix.block(1, 1, 4, 4) = math::Matrix<F>::Identity(4, 4);
  auto sparse_mds_matrix = SparseMDSMatrix<F>::FromMDSMatrix(matrix);
  EXPECT_EQ(sparse_mds_matrix.row(), matrix.row(0).transpose());
  EXPECT_EQ(sparse_mds_matrix.col_hat(), matrix.block(1, 0, 4, 1));
}

TEST_F(SparseMDSMatrixTest, Apply) {
  SparseMDSMatrix<F> sparse_mds_matrix(math::Vector<F>::Random(5),
                                       math::Vector<F>::Random(4));
  math::Matrix<F> matrix = sparse_mds_matrix.Construct();
  math::Vector<F> state = math::Vector<F>::Random(5);
  math::Vector<F> state2 = state;
  sparse_mds_matrix.Apply(state2);
  EXPECT_EQ(matrix * state, state2);
}

TEST_F(SparseMDSMatrixTest, Copyable) {
  SparseMDSMatrix<F> expected(math::Vector<F>::Random(5),
                              math::Vector<F>::Random(4));

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  SparseMDSMatrix<F> value;
  ASSERT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(value, expected);
}

}  // namespace tachyon::crypto

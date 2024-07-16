// Copyright 2022 Ethereum Foundation
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.EF and the LICENCE-APACHE.EF
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_SPARSE_MDS_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_SPARSE_MDS_MATRIX_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

// +------+------+-----+------+   +-----+
// | M₀,₀ | M₀,₁ | ... | M₀,ₜ | * | v₀  |
// +------+------+-----+------+   +-----+
// | M₁,₀ |                   |   | v₁  |
// +------+                   +   +-----+
// | ...  |         I         |   | ... |
// +------+                   +   +-----+
// | Mₜ,₀ |                   |   | vₜ  |
// +------+------+-----+------+   +-----+
//
// |v[0]| = M₀,₀ * v₀ + M₀,₁ * v₁ + ... + M₀,ₜ * vₜ
// |v[i]| = Mᵢ,₀ * v₀ + vᵢ, where 0 < i ≤ t
template <typename F>
class SparseMDSMatrix {
 public:
  SparseMDSMatrix() = default;
  SparseMDSMatrix(const math::Vector<F>& row, const math::Vector<F>& col_hat)
      : row_(row), col_hat_(col_hat) {
    CHECK_EQ(row_.size(), col_hat_.size() + 1);
  }
  SparseMDSMatrix(math::Vector<F>&& row, math::Vector<F>&& col_hat)
      : row_(std::move(row)), col_hat_(std::move(col_hat)) {
    CHECK_EQ(row_.size(), col_hat_.size() + 1);
  }

  static SparseMDSMatrix FromMDSMatrix(const math::Matrix<F>& mds_matrix) {
    CHECK_EQ(mds_matrix.rows(), mds_matrix.cols());
    CHECK_GT(mds_matrix.rows(), 1);
    CHECK_EQ(
        mds_matrix.block(1, 1, mds_matrix.rows() - 1, mds_matrix.cols() - 1),
        math::Matrix<F>::Identity(mds_matrix.rows() - 1,
                                  mds_matrix.cols() - 1));
    return {mds_matrix.row(0),
            mds_matrix.block(1, 0, mds_matrix.rows() - 1, 1)};
  }

  const math::Vector<F>& row() const { return row_; }
  const math::Vector<F>& col_hat() const { return col_hat_; }

  bool operator==(const SparseMDSMatrix& other) const {
    return row_ == other.row_ && col_hat_ == other.col_hat_;
  }
  bool operator!=(const SparseMDSMatrix& other) const {
    return !operator==(other);
  }

  void Apply(math::Vector<F>& v) const {
    F v_0 = F::Zero();
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      v_0 += row_[i] * v[i];
    }

    for (Eigen::Index i = 1; i < v.size(); ++i) {
      v[i] += col_hat_[i - 1] * v[0];
    }

    v[0] = std::move(v_0);
  }

  math::Matrix<F> Construct() const {
    math::Matrix<F> ret = math::Matrix<F>::Identity(row_.size(), row_.size());
    ret.row(0) = row_;
    ret.block(1, 0, col_hat_.size(), 1) = col_hat_;
    return ret;
  }

  std::string ToString() const {
    return absl::Substitute("{row: $0, col_hat: $1}",
                            base::ContainerToString(row_),
                            base::ContainerToString(col_hat_));
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("{row: $0, col_hat: $1}",
                            base::ContainerToHexString(row_, pad_zero),
                            base::ContainerToHexString(col_hat_, pad_zero));
  }

 private:
  // |row_| = {M₀,₀, ..., M₀,ₜ}
  math::Vector<F> row_;
  // |col_hat_| = {M₁,₀, ..., Mₜ,₀}
  math::Vector<F> col_hat_;
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::SparseMDSMatrix<F>> {
 public:
  static bool WriteTo(const crypto::SparseMDSMatrix<F>& matrix,
                      Buffer* buffer) {
    return buffer->WriteMany(matrix.row(), matrix.col_hat());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SparseMDSMatrix<F>* matrix) {
    math::Vector<F> row;
    math::Vector<F> col_hat;
    if (!buffer.ReadMany(&row, &col_hat)) {
      return false;
    }

    *matrix = {std::move(row), std::move(col_hat)};
    return true;
  }

  static size_t EstimateSize(const crypto::SparseMDSMatrix<F>& matrix) {
    return base::EstimateSize(matrix.row(), matrix.col_hat());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_SPARSE_MDS_MATRIX_H_

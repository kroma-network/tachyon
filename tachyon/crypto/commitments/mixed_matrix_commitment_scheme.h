#ifndef TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_

#include <utility>
#include <vector>

#include "tachyon/base/openmp_util.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <typename Derived>
class MixedMatrixCommitmentScheme {
 public:
  using Field = typename MixedMatrixCommitmentSchemeTraits<Derived>::Field;
  using Commitment =
      typename MixedMatrixCommitmentSchemeTraits<Derived>::Commitment;

  [[nodiscard]] bool Commit(const std::vector<Field>& vector,
                            Commitment* result) {
    math::RowMajorMatrix<Field> matrix(vector.size(), 1);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < vector.size(); ++i) {
      matrix(i, 0) = vector[i];
    }
    return Commit(std::move(matrix), result);
  }

  [[nodiscard]] bool Commit(math::RowMajorMatrix<Field>&& matrix,
                            Commitment* result) {
    return Commit(std::vector<math::RowMajorMatrix<Field>>{std::move(matrix)},
                  result);
  }

  [[nodiscard]] bool Commit(std::vector<math::RowMajorMatrix<Field>>&& matrices,
                            Commitment* result) {
    Derived* derived = static_cast<Derived*>(this);
    return derived->DoCommit(std::move(matrices), result);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_

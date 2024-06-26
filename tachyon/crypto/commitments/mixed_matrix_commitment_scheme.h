#ifndef TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme_traits_forward.h"
#include "tachyon/math/geometry/dimensions.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <typename Derived>
class MixedMatrixCommitmentScheme {
 public:
  using Field = typename MixedMatrixCommitmentSchemeTraits<Derived>::Field;
  using Commitment =
      typename MixedMatrixCommitmentSchemeTraits<Derived>::Commitment;
  using Proof = typename MixedMatrixCommitmentSchemeTraits<Derived>::Proof;

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

  [[nodiscard]] bool CreateOpeningProof(
      size_t index, std::vector<std::vector<Field>>* openings,
      Proof* proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCreateOpeningProof(index, openings, proof);
  }

  const std::vector<math::RowMajorMatrix<Field>>& GetMatrices() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoGetMatrices();
  }

  std::vector<size_t> GetRowSizes() const {
    return base::Map(GetMatrices(),
                     [](const math::RowMajorMatrix<Field>& matrix) {
                       return matrix.rows();
                     });
  }

  size_t GetMaxRowSize() const {
    const std::vector<math::RowMajorMatrix<Field>>& matrices = GetMatrices();
    if (matrices.empty()) return 0;
    return std::max_element(matrices.begin(), matrices.end(),
                            [](const math::RowMajorMatrix<Field>& a,
                               const math::RowMajorMatrix<Field>& b) {
                              return a.rows() < b.rows();
                            })
        ->rows();
  }

  [[nodiscard]] bool VerifyOpeningProof(
      const Commitment& commitment,
      absl::Span<const math::Dimensions> dimensions_list, size_t index,
      absl::Span<const std::vector<Field>> opened_values,
      const Proof& proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoVerifyOpeningProof(commitment, dimensions_list, index,
                                         opened_values, proof);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MIXED_MATRIX_COMMITMENT_SCHEME_H_

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
  using ProverData =
      typename MixedMatrixCommitmentSchemeTraits<Derived>::ProverData;
  using Proof = typename MixedMatrixCommitmentSchemeTraits<Derived>::Proof;

  [[nodiscard]] bool Commit(const std::vector<Field>& vector,
                            Commitment* commitment, ProverData* prover_data) {
    math::RowMajorMatrix<Field> matrix(vector.size(), 1);
    OMP_PARALLEL_FOR(size_t i = 0; i < vector.size(); ++i) {
      matrix(i, 0) = vector[i];
    }
    return Commit(std::move(matrix), commitment, prover_data);
  }

  [[nodiscard]] bool Commit(math::RowMajorMatrix<Field>&& matrix,
                            Commitment* commitment,
                            ProverData* prover_data) const {
    return Commit(std::vector<math::RowMajorMatrix<Field>>{std::move(matrix)},
                  commitment, prover_data);
  }

  [[nodiscard]] bool Commit(std::vector<math::RowMajorMatrix<Field>>&& matrices,
                            Commitment* commitment,
                            ProverData* prover_data) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(std::move(matrices), commitment, prover_data);
  }

  [[nodiscard]] bool CreateOpeningProof(
      size_t index, const ProverData& prover_data,
      std::vector<std::vector<Field>>* openings, Proof* proof) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCreateOpeningProof(index, prover_data, openings, proof);
  }

  const std::vector<math::RowMajorMatrix<Field>>& GetMatrices(
      const ProverData& prover_data) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoGetMatrices(prover_data);
  }

  std::vector<size_t> GetRowSizes() const {
    return base::Map(GetMatrices(),
                     [](const math::RowMajorMatrix<Field>& matrix) {
                       return matrix.rows();
                     });
  }

  size_t GetMaxRowSize(const ProverData& prover_data) const {
    const std::vector<math::RowMajorMatrix<Field>>& matrices =
        GetMatrices(prover_data);
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

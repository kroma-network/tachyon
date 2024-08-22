#ifndef TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PCS_IMPL_H_
#define TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PCS_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/crypto/commitments/fri/two_adic_fri_pcs.h"

namespace tachyon::c::crypto {

template <typename ExtF, typename InputMMCS, typename ChallengeMMCS,
          typename Challenger, typename Coset>
class TwoAdicFriPCSImpl
    : public tachyon::crypto::TwoAdicFriPCS<ExtF, InputMMCS, ChallengeMMCS,
                                            Challenger, Coset> {
 public:
  using Base = tachyon::crypto::TwoAdicFriPCS<ExtF, InputMMCS, ChallengeMMCS,
                                              Challenger, Coset>;
  using F = typename Base::F;
  using Commitment = typename Base::Commitment;
  using ProverData = typename Base::ProverData;

  using Base::Base;

  void AllocateLDEs(size_t size) { this->ldes_.reserve(size); }

  template <typename Derived>
  absl::Span<F> CosetLDEBatch(Eigen::MatrixBase<Derived>& matrix, F shift) {
    Coset coset = this->GetNaturalDomainForDegree(matrix.rows());
    tachyon::math::RowMajorMatrix<F> mat =
        coset.domain()->CosetLDEBatch(matrix, this->fri_.log_blowup, shift);
    ReverseMatrixIndexBits(mat);
    absl::Span<F> ret(mat.data(), mat.size());
    this->ldes_.push_back(std::move(mat));
    return ret;
  }

  using Base::Commit;

  void Commit(Commitment* commitment, ProverData** prover_data_out) {
    std::unique_ptr<ProverData> prover_data(new ProverData);
    CHECK(this->mmcs_.Commit(std::move(ldes_), commitment, prover_data.get()));
    *prover_data_out = prover_data.get();
    prover_data_by_round_.push_back(std::move(prover_data));
  }

 protected:
  std::vector<math::RowMajorMatrix<F>> ldes_;
  std::vector<std::unique_ptr<ProverData>> prover_data_by_round_;
};

}  // namespace tachyon::c::crypto

#endif  // TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_PCS_IMPL_H_

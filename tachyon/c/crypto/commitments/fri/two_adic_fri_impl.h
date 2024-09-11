#ifndef TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_
#define TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/commitments/fri/two_adic_fri.h"

namespace tachyon::c::crypto {

template <typename ExtF, typename InputMMCS, typename ChallengeMMCS,
          typename Challenger>
class TwoAdicFRIImpl
    : public tachyon::crypto::TwoAdicFRI<ExtF, InputMMCS, ChallengeMMCS,
                                         Challenger> {
 public:
  using Base =
      tachyon::crypto::TwoAdicFRI<ExtF, InputMMCS, ChallengeMMCS, Challenger>;
  using F = typename Base::F;
  using Commitment = typename Base::Commitment;
  using ProverData = typename Base::ProverData;
  using Domain = typename Base::Domain;

  using Base::Base;

  void AllocateLDEs(size_t size) { ldes_.reserve(size); }

  void CosetLDEBatch(Eigen::Map<tachyon::math::RowMajorMatrix<F>>&& matrix,
                     F shift,
                     Eigen::Map<tachyon::math::RowMajorMatrix<F>>& lde) {
    Domain coset = this->GetNaturalDomainForDegree(matrix.rows());
    coset.domain()->CosetLDEBatch(std::move(matrix), this->config_.log_blowup,
                                  shift, lde,
                                  /*reverse_at_last=*/false);
    ldes_.push_back(Eigen::Map<const tachyon::math::RowMajorMatrix<F>>(
        lde.data(), lde.rows(), lde.cols()));
  }

  using Base::Commit;

  void Commit(Commitment* commitment, ProverData** prover_data_out,
              std::vector<std::unique_ptr<ProverData>>* prover_data_by_round) {
    std::unique_ptr<ProverData> prover_data(new ProverData);
    CHECK(this->mmcs_.Commit(std::move(ldes_), commitment, prover_data.get()));
    *prover_data_out = prover_data.get();
    prover_data_by_round->push_back(std::move(prover_data));
  }

 protected:
  std::vector<Eigen::Map<const tachyon::math::RowMajorMatrix<F>>> ldes_;
};

}  // namespace tachyon::c::crypto

#endif  // TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

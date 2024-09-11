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
  using OpeningPoints = typename Base::OpeningPoints;
  using OpenedValues = typename Base::OpenedValues;
  using FRIProof = typename Base::FRIProof;

  using Base::Base;

  void AllocateLDEs(size_t size) { ldes_.reserve(size); }

  template <typename Derived>
  absl::Span<F> CosetLDEBatch(Eigen::MatrixBase<Derived>&& matrix, F shift) {
    Domain coset = this->GetNaturalDomainForDegree(matrix.rows());
    tachyon::math::RowMajorMatrix<F> lde(matrix.rows() << this->fri_.log_blowup,
                                         matrix.cols());
    coset.domain()->CosetLDEBatch(std::move(matrix), this->fri_.log_blowup,
                                  shift, lde,
                                  /*reverse_at_last=*/false);
    absl::Span<F> ret(lde.data(), lde.size());
    ldes_.push_back(std::move(lde));
    return ret;
  }

  using Base::Commit;

  void Commit(Commitment* commitment, ProverData** prover_data_out,
              std::vector<std::unique_ptr<ProverData>>* prover_data_by_round) {
    std::unique_ptr<ProverData> prover_data(new ProverData);
    CHECK(this->mmcs_.CommitOwned(std::move(ldes_), commitment,
                                  prover_data.get()));
    *prover_data_out = prover_data.get();
    prover_data_by_round->push_back(std::move(prover_data));
  }

 protected:
  std::vector<math::RowMajorMatrix<F>> ldes_;
};

}  // namespace tachyon::c::crypto

#endif  // TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

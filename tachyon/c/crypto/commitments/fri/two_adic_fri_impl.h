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

  void AllocateLDEs(size_t size) { this->ldes_.reserve(size); }

  template <typename Derived>
  absl::Span<F> CosetLDEBatch(Eigen::MatrixBase<Derived>& matrix, F shift) {
    Domain coset = this->GetNaturalDomainForDegree(matrix.rows());
    tachyon::math::RowMajorMatrix<F> mat = coset.domain()->CosetLDEBatch(
        matrix, this->fri_.log_blowup, shift, /*reverse_at_last=*/false);
    absl::Span<F> ret(mat.data(), mat.size());
    this->ldes_.push_back(std::move(mat));
    return ret;
  }

  using Base::Commit;

  void Commit(Commitment* commitment, ProverData** prover_data_out,
              std::vector<std::unique_ptr<ProverData>>* prover_data_by_round) {
    std::unique_ptr<ProverData> prover_data(new ProverData);
    CHECK(this->mmcs_.Commit(std::move(ldes_), commitment, prover_data.get()));
    *prover_data_out = prover_data.get();
    prover_data_by_round->push_back(std::move(prover_data));
  }

  void CreateOpeningProof(
      const std::vector<std::unique_ptr<ProverData>>& prover_data_by_round_in,
      const OpeningPoints& points_by_round, Challenger& challenger,
      OpenedValues* opened_values_by_round, FRIProof* proof) const {
    auto& prover_data_by_round =
        const_cast<std::vector<std::unique_ptr<ProverData>>&>(
            prover_data_by_round_in);
    std::vector<ProverData> prover_data_by_round_tmp = tachyon::base::Map(
        prover_data_by_round, [](std::unique_ptr<ProverData>& prover_data) {
          return ProverData(std::move(*prover_data));
        });
    CHECK(Base::CreateOpeningProof(prover_data_by_round_tmp, points_by_round,
                                   challenger, opened_values_by_round, proof));
    prover_data_by_round = tachyon::base::Map(
        prover_data_by_round_tmp, [](ProverData& prover_data) {
          return std::make_unique<ProverData>(std::move(prover_data));
        });
  }

 protected:
  std::vector<math::RowMajorMatrix<F>> ldes_;
};

}  // namespace tachyon::c::crypto

#endif  // TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

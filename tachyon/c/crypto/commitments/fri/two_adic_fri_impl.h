#ifndef TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_
#define TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/debugging/leak_check.h"

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
  using FRIProof = typename Base::FRIProof;
  using OpeningPoints = typename Base::OpeningPoints;
  using OpenedValues = typename Base::OpenedValues;

  using Base::Base;

  void CosetLDEBatch(Eigen::Map<tachyon::math::RowMajorMatrix<F>>&& matrix,
                     F shift,
                     Eigen::Map<tachyon::math::RowMajorMatrix<F>>& lde) {
    Domain coset = this->GetNaturalDomainForDegree(matrix.rows());
    coset.domain()->CosetLDEBatch(std::move(matrix), this->config_.log_blowup,
                                  shift, lde,
                                  /*reverse_at_last=*/false);
  }

  using Base::Commit;

  void Commit(
      std::vector<Eigen::Map<const tachyon::math::RowMajorMatrix<F>>>&& ldes,
      Commitment* commitment, ProverData** prover_data_out) {
    // NOTE(chokobole): The caller is responsible for deallocating the memory.
    // |TwoAdicFri| in Plonky3 is stateless, allowing it to be used in
    // multithreaded contexts. As a result, the newly created |ProverData|
    // object cannot be owned by the |TwoAdicFri| instance itself.
    *prover_data_out = absl::IgnoreLeak(new ProverData());
    CHECK(this->mmcs_.Commit(std::move(ldes), commitment, *prover_data_out));
  }

  void CreateOpeningProof(
      const std::vector<const ProverData*>& prover_data_by_round,
      const OpeningPoints& points_by_round, Challenger& challenger,
      OpenedValues* opened_values_out, FRIProof* proof) const {
    std::vector<std::unique_ptr<ProverData>> prover_data_by_round_tmp =
        tachyon::base::Map(prover_data_by_round, [](const ProverData* data) {
          return std::unique_ptr<ProverData>(const_cast<ProverData*>(data));
        });
    CHECK(Base::CreateOpeningProof(prover_data_by_round_tmp, points_by_round,
                                   challenger, opened_values_out, proof));
    for (std::unique_ptr<ProverData>& prover_data : prover_data_by_round_tmp) {
      // NOTE(chokobole): The caller is responsible for deallocating the memory.
      // See the comment in |Commit()|.
      absl::IgnoreLeak(prover_data.release());
    }
  }
};

}  // namespace tachyon::c::crypto

#endif  // TACHYON_C_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_IMPL_H_

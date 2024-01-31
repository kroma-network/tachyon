#ifndef TACHYON_C_ZK_PLONK_HALO2_PROVER_IMPL_BASE_H_
#define TACHYON_C_ZK_PLONK_HALO2_PROVER_IMPL_BASE_H_

#include <memory>
#include <utility>

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/halo2/prover.h"

namespace tachyon::c::zk::plonk::halo2 {

template <typename PCS>
class ProverImplBase : public tachyon::zk::plonk::halo2::Prover<PCS> {
 public:
  using Poly = typename PCS::Poly;
  using Evals = typename PCS::Evals;
  using Commitment = typename PCS::Commitment;

  explicit ProverImplBase(
      base::OnceCallback<tachyon::zk::plonk::halo2::Prover<PCS>()> callback)
      : tachyon::zk::plonk::halo2::Prover<PCS>(std::move(callback).Run()) {}

  void SetRng(std::unique_ptr<crypto::XORShiftRNG> rng) {
    tachyon::zk::plonk::halo2::Prover<PCS>::SetRng(std::move(rng));
  }

  void SetTranscript(
      std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer) {
    this->transcript_ = std::move(writer);
  }

  void CreateProof(
      const tachyon::zk::plonk::ProvingKey<Poly, Evals, Commitment>&
          proving_key,
      tachyon::zk::plonk::halo2::ArgumentData<Poly, Evals>* argument_data) {
    tachyon::zk::plonk::halo2::Prover<PCS>::CreateProof(proving_key,
                                                        argument_data);
  }
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_PROVER_IMPL_BASE_H_

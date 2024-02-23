// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_
#define TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/blinder.h"
#include "tachyon/zk/base/entities/entity.h"

namespace tachyon::zk {

template <typename PCS>
class ProverBase : public Entity<PCS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;
  using Poly = typename PCS::Poly;
  using Commitment = typename PCS::Commitment;

  ProverBase(PCS&& pcs,
             std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
             Blinder<F>&& blinder)
      : Entity<PCS>(std::move(pcs), std::move(writer)),
        blinder_(std::move(blinder)) {}

  Blinder<F>& blinder() { return blinder_; }

  crypto::TranscriptWriter<Commitment>* GetWriter() {
    return this->transcript()->ToWriter();
  }
  const crypto::TranscriptWriter<Commitment>* GetWriter() const {
    return this->transcript()->ToWriter();
  }

  RowIndex GetUsableRows() const {
    return Entity<PCS>::GetUsableRows(blinder_.blinding_factors());
  }

  void CommitAndWriteToTranscript(const Poly& poly) {
    Commitment commitment = this->Commit(poly);
    CHECK(GetWriter()->WriteToTranscript(commitment));
  }

  void CommitAndWriteToProof(const Poly& poly) {
    Commitment commitment = this->Commit(poly);
    CHECK(GetWriter()->WriteToProof(commitment));
  }

  void CommitAndWriteToTranscript(const Evals& evals) {
    Commitment commitment = this->Commit(evals);
    CHECK(GetWriter()->WriteToTranscript(commitment));
  }

  void CommitAndWriteToProof(const Evals& evals) {
    Commitment commitment = this->Commit(evals);
    CHECK(GetWriter()->WriteToProof(commitment));
  }

  template <typename Container>
  void CommitAndWriteToTranscript(const Container& container) {
    Commitment commitment = this->Commit(container);
    CHECK(GetWriter()->WriteToTranscript(commitment));
  }

  template <typename Container>
  void CommitAndWriteToProof(const Container& container) {
    Commitment commitment = this->Commit(container);
    CHECK(GetWriter()->WriteToProof(commitment));
  }

  void EvaluateAndWriteToProof(const Poly& poly, const F& x) {
    F result = poly.Evaluate(x);
    CHECK(GetWriter()->WriteToProof(result));
  }

  template <typename T = PCS,
            std::enable_if_t<crypto::VectorCommitmentSchemeTraits<
                T>::kSupportsBatchMode>* = nullptr>
  void RetrieveAndWriteBatchCommitmentsToProof() {
    std::vector<Commitment> commitments = this->pcs_.GetBatchCommitments();
    for (const Commitment& commitment : commitments) {
      CHECK(GetWriter()->WriteToProof(commitment));
    }
  }

  template <typename T = PCS,
            std::enable_if_t<crypto::VectorCommitmentSchemeTraits<
                T>::kSupportsBatchMode>* = nullptr>
  void RetrieveAndWriteBatchCommitmentsToTranscript() {
    std::vector<Commitment> commitments = this->pcs_.GetBatchCommitments();
    for (const Commitment& commitment : commitments) {
      CHECK(GetWriter()->WriteToTranscript(commitment));
    }
  }

 protected:
  Blinder<F> blinder_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_

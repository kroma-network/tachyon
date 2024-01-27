// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include "openssl/blake2.h"

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/keys/key.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon {
namespace halo2_api {

template <typename PCS>
class ProvingKeyImpl;

}  // namespace halo2_api

namespace zk {
namespace halo2 {

template <typename PCS>
class PinnedVerifyingKey;

}  // namespace halo2

template <typename PCS>
class ProvingKey;

template <typename PCS>
class VerifyingKey : public Key<PCS> {
 public:
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;
  using Commitment = typename PCS::Commitment;
  using Commitments = std::vector<Commitment>;
  using PreLoadResult = typename Key<PCS>::PreLoadResult;

  VerifyingKey() = default;

  const Commitments& fixed_commitments() const { return fixed_commitments_; }

  const PermutationVerifyingKey<Commitment>& permutation_verifying_key() const {
    return permutation_verifying_key_;
  }

  const ConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }

  const F& transcript_repr() const { return transcript_repr_; }

  // Return true if it is able to load from an instance of |circuit|.
  template <typename Circuit>
  [[nodiscard]] bool Load(Entity<PCS>* entity, const Circuit& circuit) {
    PreLoadResult result;
    if (!this->PreLoad(entity, circuit, &result)) return false;
    return DoLoad(entity, std::move(result), nullptr);
  }

 private:
  friend class ProvingKey<PCS>;
  friend class halo2_api::ProvingKeyImpl<PCS>;

  struct LoadResult {
    std::vector<Evals> permutations;
  };

  bool DoLoad(Entity<PCS>* entity, PreLoadResult&& pre_load_result,
              LoadResult* load_result) {
    constraint_system_ = std::move(pre_load_result.constraint_system);

    std::vector<Evals> permutations =
        pre_load_result.assembly.permutation().GeneratePermutations(
            entity->domain());
    permutation_verifying_key_ =
        pre_load_result.assembly.permutation().BuildVerifyingKey(entity,
                                                                 permutations);
    if (load_result) {
      load_result->permutations = std::move(permutations);
    }

    const PCS& pcs = entity->pcs();
    // TODO(chokobole): Parallelize this.
    fixed_commitments_ =
        base::Map(pre_load_result.fixed_columns, [&pcs](const Evals& evals) {
          Commitment commitment;
          CHECK(pcs.CommitLagrange(evals, &commitment));
          return commitment;
        });

    SetTranscriptRepresentative(entity);
    return true;
  }

  void SetTranscriptRepresentative(const Entity<PCS>* entity) {
    halo2::PinnedVerifyingKey<PCS> pinned_verifying_key(entity, *this);

    std::string vk_str = base::ToRustDebugString(pinned_verifying_key);
    size_t vk_str_size = vk_str.size();

    BLAKE2B_CTX state;
    BLAKE2B512_InitWithPersonal(&state, halo2::kVerifyingKeyStr);
    BLAKE2B512_Update(&state, reinterpret_cast<const uint8_t*>(&vk_str_size),
                      sizeof(size_t));
    BLAKE2B512_Update(&state, vk_str.data(), vk_str.size());
    uint8_t result[64] = {0};
    BLAKE2B512_Final(result, &state);

    transcript_repr_ =
        F::FromAnySizedBigInt(math::BigInt<8>::FromBytesLE(result));
  }

  Commitments fixed_commitments_;
  PermutationVerifyingKey<Commitment> permutation_verifying_key_;
  ConstraintSystem<F> constraint_system_;
  // The representative of this |VerifyingKey| in transcripts.
  F transcript_repr_ = F::Zero();
};

}  // namespace zk
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

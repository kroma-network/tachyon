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

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/halo2/constants.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key_forward.h"
#include "tachyon/zk/plonk/halo2/prime_field_conversion.h"
#include "tachyon/zk/plonk/keys/c_proving_key_impl_forward.h"
#include "tachyon/zk/plonk/keys/key.h"
#include "tachyon/zk/plonk/keys/proving_key_forward.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon::zk::plonk {

template <typename Evals>
struct VerifyingKeyLoadResult {
  std::vector<Evals> permutations;
};

template <typename F, typename C>
class VerifyingKey : public Key {
 public:
  using Commitments = std::vector<C>;

  VerifyingKey() = default;

  const Commitments& fixed_commitments() const { return fixed_commitments_; }

  const PermutationVerifyingKey<C>& permutation_verifying_key() const {
    return permutation_verifying_key_;
  }

  const ConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }

  const F& transcript_repr() const { return transcript_repr_; }

  void SetTranscriptReprForTesting(const F& transcript_repr) {
    transcript_repr_ = transcript_repr;
  }

  // Return true if it is able to load from an instance of |circuit|.
  template <typename PCS, typename Circuit>
  [[nodiscard]] bool Load(Entity<PCS>* entity, const Circuit& circuit,
                          lookup::Type lookup_type) {
    using Evals = typename PCS::Evals;
    using RationalEvals = typename PCS::RationalEvals;
    KeyPreLoadResult<Evals, RationalEvals> result(lookup_type);
    if (!this->PreLoad(entity, circuit, &result)) return false;
    // NOTE(chokobole): This helps the compiler to infer type easily.
    // Without this line, it won't compile.
    VerifyingKeyLoadResult<Evals>* load_result = nullptr;
    return DoLoad(entity, std::move(result), load_result);
  }

 private:
  template <halo2::Vendor, typename LS>
  friend class ProvingKey;
  template <tachyon::zk::plonk::halo2::Vendor Vendor, typename LS>
  friend class c::zk::plonk::ProvingKeyImpl;

  template <typename PCS, typename Evals, typename RationalEvals>
  bool DoLoad(Entity<PCS>* entity,
              KeyPreLoadResult<Evals, RationalEvals>&& pre_load_result,
              VerifyingKeyLoadResult<Evals>* load_result) {
    constraint_system_ = std::move(pre_load_result.constraint_system);

    std::vector<Evals> permutations =
        pre_load_result.assembly.permutation()
            .template GeneratePermutations<Evals>(entity->domain());
    permutation_verifying_key_ =
        pre_load_result.assembly.permutation().BuildVerifyingKey(entity,
                                                                 permutations);
    if (load_result) {
      load_result->permutations = std::move(permutations);
    }

    const PCS& pcs = entity->pcs();
    fixed_commitments_ =
        base::Map(pre_load_result.fixed_columns, [&pcs](const Evals& evals) {
          C commitment;
          CHECK(pcs.CommitLagrange(evals, &commitment));
          return commitment;
        });

    SetTranscriptRepresentative(entity);
    return true;
  }

  template <typename PCS>
  void SetTranscriptRepresentative(const Entity<PCS>* entity) {
    halo2::PinnedVerifyingKey<F, C> pinned_verifying_key(entity, *this);

    std::string vk_str = base::ToRustDebugString(pinned_verifying_key);
    VLOG(3) << vk_str;
    size_t vk_str_size = vk_str.size();

    BLAKE2B_CTX state;
    BLAKE2B512_InitWithPersonal(&state, halo2::kVerifyingKeyStr);
    BLAKE2B512_Update(&state, reinterpret_cast<const uint8_t*>(&vk_str_size),
                      sizeof(size_t));
    BLAKE2B512_Update(&state, vk_str.data(), vk_str.size());
    uint8_t result[BLAKE2B512_DIGEST_LENGTH] = {0};
    BLAKE2B512_Final(result, &state);

    transcript_repr_ = halo2::FromUint512<F>(result);
    VLOG(2) << "Halo2(transcript_repr): " << transcript_repr_.ToHexString(true);
  }

  Commitments fixed_commitments_;
  PermutationVerifyingKey<C> permutation_verifying_key_;
  ConstraintSystem<F> constraint_system_;
  // The representative of this |VerifyingKey| in transcripts.
  F transcript_repr_ = F::Zero();
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

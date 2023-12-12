// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openssl/blake2.h"

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/keys/assembly.h"
#include "tachyon/zk/plonk/permutation/permutation_verifying_key.h"

namespace tachyon::zk {
namespace halo2 {

template <typename PCSTy>
class PinnedVerifyingKey;

}  // namespace halo2

constexpr char kVerifyingKeyStr[] = "Halo2-Verify-Key";

template <typename PCSTy>
class VerifyingKey {
 public:
  using F = typename PCSTy::Field;
  using Commitment = typename PCSTy::Commitment;
  using Commitments = std::vector<Commitment>;

  VerifyingKey() = default;
  VerifyingKey(Commitments&& fixed_commitments,
               PermutationVerifyingKey<PCSTy>&& permutation_verifying_key,
               ConstraintSystem<F>&& constraint_system)
      : fixed_commitments_(std::move(fixed_commitments)),
        permutation_verifying_Key_(std::move(permutation_verifying_key)),
        constraint_system_(std::move(constraint_system)) {}

  static Assembly<PCSTy> CreateAssembly(
      const PCSTy& pcs, const ConstraintSystem<F>& constraint_system) {
    using RationalEvals = typename Assembly<PCSTy>::RationalEvals;
    return {
        static_cast<uint32_t>(pcs.K()),
        base::CreateVector(constraint_system.num_fixed_columns(),
                           RationalEvals::UnsafeZero(pcs.N() - 1)),
        PermutationAssembly<PCSTy>(constraint_system.permutation(), pcs.N()),
        base::CreateVector(constraint_system.num_selectors(),
                           base::CreateVector(pcs.N(), false)),
        base::Range<size_t>::Until(
            pcs.N() - (constraint_system.ComputeBlindingFactors() + 1))};
  }

  static VerifyingKey FromParts(
      const Entity<PCSTy>* entity, Commitments fixed_commitments,
      PermutationVerifyingKey<PCSTy> permutation_verifying_key,
      ConstraintSystem<F> constraint_system) {
    VerifyingKey ret(std::move(fixed_commitments),
                     std::move(permutation_verifying_key),
                     std::move(constraint_system));
    ret.SetTranscriptRepresentative(entity);
    return ret;
  }

  template <typename CircuitTy>
  [[nodiscard]] static bool Generate(Entity<PCSTy>* entity,
                                     const CircuitTy& circuit,
                                     VerifyingKey* verifying_key);

  const Commitments& fixed_commitments() const { return fixed_commitments_; }

  const PermutationVerifyingKey<PCSTy>& permutation_verifying_key() const {
    return permutation_verifying_Key_;
  }

  const ConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }

  const F& transcript_repr() const { return transcript_repr_; }

 private:
  void SetTranscriptRepresentative(const Entity<PCSTy>* entity) {
    halo2::PinnedVerifyingKey<PCSTy> pinned_verifying_key(entity, *this);

    std::string vk_str = base::ToRustDebugString(pinned_verifying_key);
    size_t vk_str_size = vk_str.size();

    BLAKE2B_CTX state;
    BLAKE2B512_InitWithPersonal(&state, kVerifyingKeyStr);
    BLAKE2B512_Update(&state, reinterpret_cast<const uint8_t*>(&vk_str_size),
                      sizeof(size_t));
    BLAKE2B512_Update(&state, vk_str.data(), vk_str.size());
    uint8_t result[64] = {0};
    BLAKE2B512_Final(result, &state);

    transcript_repr_ =
        F::FromAnySizedBigInt(math::BigInt<8>::FromBytesLE(result));
  }

  Commitments fixed_commitments_;
  PermutationVerifyingKey<PCSTy> permutation_verifying_Key_;
  ConstraintSystem<F> constraint_system_;
  // The representative of this |VerifyingKey| in transcripts.
  F transcript_repr_ = F::Zero();
};

// static
template <typename PCSTy>
template <typename CircuitTy>
bool VerifyingKey<PCSTy>::Generate(Entity<PCSTy>* entity,
                                   const CircuitTy& circuit,
                                   VerifyingKey* verifying_key) {
  using Config = typename CircuitTy::Config;
  using FloorPlanner = typename CircuitTy::FloorPlanner;
  using RationalEvals = typename Assembly<PCSTy>::RationalEvals;
  using Evals = typename PCSTy::Evals;
  using ExtendedDomain = typename PCSTy::ExtendedDomain;

  ConstraintSystem<F> constraint_system;
  Config config = CircuitTy::Configure(constraint_system);

  PCSTy& pcs = entity->pcs();
  if (pcs.N() < constraint_system.ComputeMinimumRows()) {
    LOG(ERROR) << "Not enough rows available " << pcs.N() << " vs "
               << constraint_system.ComputeMinimumRows();
    return false;
  }
  size_t extended_k = constraint_system.ComputeExtendedDegree(pcs.K());
  entity->set_extended_domain(ExtendedDomain::Create(size_t{1} << extended_k));

  Assembly<PCSTy> assembly = CreateAssembly(pcs, constraint_system);
  FloorPlanner::Synthesize(&assembly, circuit, std::move(config),
                           constraint_system.constants());

  std::vector<Evals> fixed_columns =
      base::Map(assembly.fixed_columns(), [](const RationalEvals& evals) {
        std::vector<F> result =
            base::CreateVector(evals.evaluations().size(), F::Zero());
        CHECK(math::RationalField<F>::BatchEvaluate(evals.evaluations(),
                                                    &result));
        return Evals(std::move(result));
      });

  std::vector<std::vector<F>> selector_polys_tmp =
      constraint_system.CompressSelectors(assembly.selectors());
  std::vector<Evals> selector_polys =
      base::Map(std::make_move_iterator(selector_polys_tmp.begin()),
                std::make_move_iterator(selector_polys_tmp.end()),
                [](std::vector<F>&& vec) { return Evals(std::move(vec)); });
  fixed_columns.insert(fixed_columns.end(),
                       std::make_move_iterator(selector_polys.begin()),
                       std::make_move_iterator(selector_polys.end()));

  std::vector<Evals> permutations =
      assembly.permutation().GeneratePermutations(entity->domain());
  PermutationVerifyingKey<PCSTy> permutation_vk =
      assembly.permutation().BuildVerifyingKey(entity, permutations);

  // TODO(chokobole): Parallelize this.
  Commitments fixed_commitments =
      base::Map(fixed_columns, [&pcs](const Evals& evals) {
        Commitment commitment;
        CHECK(pcs.CommitLagrange(evals, &commitment));
        return commitment;
      });

  *verifying_key = VerifyingKey::FromParts(entity, std::move(fixed_commitments),
                                           std::move(permutation_vk),
                                           std::move(constraint_system));
  return true;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_VERIFYING_KEY_H_

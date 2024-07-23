// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PINNED_VERIFYING_KEY_H_
#define TACHYON_ZK_PLONK_HALO2_PINNED_VERIFYING_KEY_H_

#include <memory>
#include <string>
#include <vector>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/plonk/halo2/pinned_constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_evaluation_domain.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key_forward.h"
#include "tachyon/zk/plonk/halo2/stringifiers/permutation_verifying_key_stringifier.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

namespace tachyon {
namespace zk::plonk::halo2 {

template <typename F, typename C>
class PinnedVerifyingKey {
 public:
  using BaseField = typename C::BaseField;
  using ScalarField = typename C::ScalarField;

  template <typename PCS>
  PinnedVerifyingKey(const Entity<PCS>* entity, const VerifyingKey<F, C>& vk)
      : base_modulus_(BaseField::Config::kModulus.ToHexString(true)),
        scalar_modulus_(ScalarField::Config::kModulus.ToHexString(true)),
        domain_(entity),
        constraint_system_(vk.constraint_system()),
        fixed_commitments_(vk.fixed_commitments()),
        permutation_verifying_key_(vk.permutation_verifying_key()) {}

  const std::string& base_modulus() const { return base_modulus_; }
  const std::string& scalar_modulus() const { return scalar_modulus_; }
  const PinnedEvaluationDomain<F>& domain() const { return domain_; }
  const PinnedConstraintSystem<F>& constraint_system() const {
    return constraint_system_;
  }
  const std::vector<C>& fixed_commitments() const { return fixed_commitments_; }
  const PermutationVerifyingKey<C>& permutation_verifying_key() const {
    return permutation_verifying_key_;
  }

 private:
  std::string base_modulus_;
  std::string scalar_modulus_;
  PinnedEvaluationDomain<F> domain_;
  PinnedConstraintSystem<F> constraint_system_;
  const std::vector<C>& fixed_commitments_;
  const PermutationVerifyingKey<C>& permutation_verifying_key_;
};

}  // namespace zk::plonk::halo2

namespace base::internal {

template <typename F, typename C>
class RustDebugStringifier<zk::plonk::halo2::PinnedVerifyingKey<F, C>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::plonk::halo2::PinnedVerifyingKey<F, C>& pinned_vk) {
    // NOTE(chokobole): Original name is PinnedVerificationKey not
    // PinnedVerifyingKey. See
    // https://github.com/kroma-network/halo2/blob/7d0a369/halo2_proofs/src/plonk.rs#L252-L263.
    return os << fmt.DebugStruct("PinnedVerificationKey")
                     .Field("base_modulus", pinned_vk.base_modulus())
                     .Field("scalar_modulus", pinned_vk.scalar_modulus())
                     .Field("domain", pinned_vk.domain())
                     .Field("cs", pinned_vk.constraint_system())
                     .Field("fixed_commitments", pinned_vk.fixed_commitments())
                     .Field("permutation",
                            pinned_vk.permutation_verifying_key())
                     .Finish();
  }
};

}  // namespace base::internal
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PINNED_VERIFYING_KEY_H_

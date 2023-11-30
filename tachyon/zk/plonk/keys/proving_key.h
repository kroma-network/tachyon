// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/keys/verifying_key.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"
#include "tachyon/zk/plonk/vanishing/cl_evaluator.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProvingKey {
 public:
  static constexpr size_t kMaxDegree = PCSTy::kMaxDegree;

  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;
  using Evals = typename PCSTy::Evals;

  ProvingKey() = default;
  ProvingKey(VerifyingKey<PCSTy> verifying_key, Poly l0, Poly l_last,
             Poly l_active_row, Evals fixed_values, Evals fixed_polys,
             PermutationProvingKey<PCSTy> permutation_proving_key)
      : verifying_key_(std::move(verifying_key)),
        l0_(std::move(l0)),
        l_last_(std::move(l_last)),
        l_active_row_(std::move(l_active_row)),
        fixed_values_(std::move(fixed_values)),
        fixed_polys_(std::move(fixed_polys)),
        permutation_proving_key_(std::move(permutation_proving_key)) {}

  const VerifyingKey<PCSTy>& verifying_key() const { return verifying_key_; }
  const Poly& l0() const { return l0_; }
  const Poly& l_last() const { return l_last_; }
  const Poly& l_active_row() const { return l_active_row_; }
  const std::vector<Evals>& fixed_values() const { return fixed_values_; }
  const std::vector<Evals>& fixed_polys() const { return fixed_polys_; }
  const PermutationProvingKey<PCSTy>& permutation_proving_key() const {
    return permutation_proving_key_;
  }

 private:
  VerifyingKey<PCSTy> verifying_key_;
  Poly l0_;
  Poly l_last_;
  Poly l_active_row_;
  std::vector<Evals> fixed_values_;
  std::vector<Evals> fixed_polys_;
  PermutationProvingKey<PCSTy> permutation_proving_key_;
  CLEvaluator<PCSTy> evaluator_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_KEYS_PROVING_KEY_H_

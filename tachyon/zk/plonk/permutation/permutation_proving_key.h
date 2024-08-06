// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/keys/c_proving_key_impl_forward.h"

namespace tachyon::zk::plonk {

template <typename Poly, typename Evals, typename ExtendedEvals>
class PermutationProvingKey {
 public:
  PermutationProvingKey() = default;
  PermutationProvingKey(const std::vector<Evals>& permutations,
                        const std::vector<Poly>& polys,
                        const std::vector<ExtendedEvals>& cosets)
      : permutations_(permutations), polys_(polys), cosets_(cosets) {}
  PermutationProvingKey(std::vector<Evals>&& permutations,
                        std::vector<Poly>&& polys,
                        std::vector<ExtendedEvals>&& cosets)
      : permutations_(std::move(permutations)),
        polys_(std::move(polys)),
        cosets_(std::move(cosets)) {}

  const std::vector<Evals>& permutations() const { return permutations_; }
  const std::vector<Poly>& polys() const { return polys_; }
  std::vector<Poly>& polys() { return polys_; }
  const std::vector<ExtendedEvals>& cosets() const { return cosets_; }

  size_t BytesLength() const { return base::EstimateSize(this); }

  bool operator==(const PermutationProvingKey& other) const {
    return permutations_ == other.permutations_ && polys_ == other.polys_ &&
           cosets_ == other.cosets_;
  }
  bool operator!=(const PermutationProvingKey& other) const {
    return !operator==(other);
  }

 private:
  template <typename PS>
  friend class c::zk::plonk::ProvingKeyImpl;

  std::vector<Evals> permutations_;
  std::vector<Poly> polys_;
  // NOTE(chokobole): Only PSE Halo2 has the member |cosets_|.
  // See below:
  // https://github.com/privacy-scaling-explorations/halo2/blob/bc857a7/halo2_backend/src/plonk/permutation.rs#L59-L63
  // https://github.com/scroll-tech/halo2/blob/1070391/halo2_proofs/src/plonk/permutation.rs#L123-L126
  std::vector<ExtendedEvals> cosets_;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_

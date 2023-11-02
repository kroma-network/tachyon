// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace zk {

template <typename PCSTy>
class PermutationProvingKey {
 public:
  constexpr static size_t kMaxDegree = PCSTy::kMaxDegree;

  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;
  using Evals = typename PCSTy::Evals;

  PermutationProvingKey() = default;
  PermutationProvingKey(const std::vector<Evals>& permutations,
                        const std::vector<Poly>& polys)
      : permutations_(permutations), polys_(polys) {}
  PermutationProvingKey(std::vector<Evals>&& permutations,
                        std::vector<Poly>&& polys)
      : permutations_(std::move(permutations)), polys_(std::move(polys)) {}

  const std::vector<Evals>& permutations() const { return permutations_; }
  const std::vector<Poly>& polys() const { return polys_; }

  size_t BytesLength() const { return base::EstimateSize(this); }

  bool operator==(const PermutationProvingKey& other) const {
    return permutations_ == other.permutations_ && polys_ == other.polys_;
  }
  bool operator!=(const PermutationProvingKey& other) const {
    return !operator==(other);
  }

 private:
  std::vector<Evals> permutations_;
  std::vector<Poly> polys_;
};

}  // namespace zk

namespace base {

template <typename PCSTy>
class Copyable<zk::PermutationProvingKey<PCSTy>> {
 public:
  using Poly = typename PCSTy::Poly;
  using Evals = typename PCSTy::Evals;

  static bool WriteTo(const zk::PermutationProvingKey<PCSTy>& pk,
                      Buffer* buffer) {
    return buffer->WriteMany(pk.permutations(), pk.polys());
  }

  static bool ReadFrom(const Buffer& buffer,
                       zk::PermutationProvingKey<PCSTy>* pk) {
    std::vector<Evals> perms;
    std::vector<Poly> poly;
    if (!buffer.ReadMany(&perms, &poly)) return false;

    *pk = zk::PermutationProvingKey<PCSTy>(std::move(perms), std::move(poly));
    return true;
  }

  static size_t EstimateSize(const zk::PermutationProvingKey<PCSTy>& pk) {
    return base::EstimateSize(pk.permutations()) +
           base::EstimateSize(pk.polys());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_

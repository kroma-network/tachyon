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
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon {
namespace zk {

template <typename F, size_t MaxDegree>
class PermutationProvingKey {
 public:
  using DensePoly = math::UnivariateDensePolynomial<F, MaxDegree>;
  using Evals = math::UnivariateEvaluations<F, MaxDegree>;

  PermutationProvingKey() = default;
  PermutationProvingKey(const std::vector<Evals>& permutations,
                        const std::vector<DensePoly>& polys)
      : permutations_(permutations), polys_(polys) {}
  PermutationProvingKey(std::vector<Evals>&& permutations,
                        std::vector<DensePoly>&& polys)
      : permutations_(std::move(permutations)), polys_(std::move(polys)) {}

  const std::vector<Evals>& permutations() const { return permutations_; }
  const std::vector<DensePoly>& polys() const { return polys_; }

  size_t BytesLength() const { return base::EstimateSize(this); }

  bool operator==(const PermutationProvingKey& other) const {
    return permutations_ == other.permutations_ && polys_ == other.polys_;
  }
  bool operator!=(const PermutationProvingKey& other) const {
    return !operator==(other);
  }

 private:
  std::vector<Evals> permutations_;
  std::vector<DensePoly> polys_;
};

}  // namespace zk

namespace base {

template <typename F, size_t MaxDegree>
class Copyable<zk::PermutationProvingKey<F, MaxDegree>> {
 public:
  static bool WriteTo(const zk::PermutationProvingKey<F, MaxDegree>& pk,
                      Buffer* buffer) {
    return buffer->WriteMany(pk.permutations(), pk.polys());
  }

  static bool ReadFrom(const Buffer& buffer,
                       zk::PermutationProvingKey<F, MaxDegree>* pk) {
    std::vector<math::UnivariateEvaluations<F, MaxDegree>> perms;
    std::vector<math::UnivariateDensePolynomial<F, MaxDegree>> poly;
    if (!buffer.ReadMany(&perms, &poly)) return false;

    *pk = zk::PermutationProvingKey<F, MaxDegree>(std::move(perms),
                                                  std::move(poly));
    return true;
  }

  static size_t EstimateSize(
      const zk::PermutationProvingKey<F, MaxDegree>& pk) {
    return base::EstimateSize(pk.permutations()) +
           base::EstimateSize(pk.polys());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_PROVING_KEY_H_

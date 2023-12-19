#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_

#include <stddef.h>

#include "tachyon/base/numerics/checked_math.h"
// TODO(chokobole): Remove this header. See comment in |GetDelta()| below.
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

constexpr size_t ComputePermutationChunkLength(size_t cs_degree) {
  base::CheckedNumeric<size_t> checked_cs_degree(cs_degree);
  return (checked_cs_degree - 2).ValueOrDie();
}

// Calculate δ = g^2ˢ with order T (i.e., T-th root of unity),
// where T = F::Config::kTrace.
template <typename F>
constexpr F GetDelta() {
  // NOTE(chokobole): The resulting value is different from the one in
  // https://github.com/kroma-network/halo2curves/blob/c0ac1935e5da2a620204b5b011be2c924b1e0155/src/bn256/fr.rs#L101-L110.
  // This is an ugly way to produce a same result with Halo2Curves but we will
  // remove once we don't have to match it against Halo2 any longer in the
  // future.
  if constexpr (std::is_same_v<F, math::bn254::Fr>) {
    return F::FromMontgomery(math::BigInt<4>(
        {UINT64_C(11100302345850292309), UINT64_C(5109383341788583484),
         UINT64_C(6450182039226333095), UINT64_C(2498166472155664813)}));
  } else {
    F g = F::FromMontgomery(F::Config::kSubgroupGenerator);
    F adicity = F(2).Pow(F::Config::kTwoAdicity);
    return g.Pow(adicity.ToBigInt());
  }
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_UTILS_H_

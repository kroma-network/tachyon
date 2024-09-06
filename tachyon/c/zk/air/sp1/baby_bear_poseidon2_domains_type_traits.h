#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_TYPE_TRAITS_H_

#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<std::vector<std::vector<
    tachyon::crypto::TwoAdicMultiplicativeCoset<tachyon::math::BabyBear>>>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_domains;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_domains> {
  using NativeType = std::vector<std::vector<
      tachyon::crypto::TwoAdicMultiplicativeCoset<tachyon::math::BabyBear>>>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DOMAINS_TYPE_TRAITS_H_

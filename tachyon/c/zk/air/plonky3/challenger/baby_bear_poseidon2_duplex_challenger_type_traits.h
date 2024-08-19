#ifndef TACHYON_C_ZK_AIR_PLONKY3_CHALLENGER_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_PLONKY3_CHALLENGER_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/zk/air/plonky3/challenger/duplex_challenger.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::zk::air::plonky3::DuplexChallenger<
    tachyon::crypto::Poseidon2Sponge<tachyon::crypto::Poseidon2ExternalMatrix<
        tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
            tachyon::math::BabyBear>>>,
    16, 8>> {
  using CType = tachyon_plonky3_baby_bear_poseidon2_duplex_challenger;
};

template <>
struct TypeTraits<tachyon_plonky3_baby_bear_poseidon2_duplex_challenger> {
  using NativeType = tachyon::zk::air::plonky3::DuplexChallenger<
      tachyon::crypto::Poseidon2Sponge<tachyon::crypto::Poseidon2ExternalMatrix<
          tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
              tachyon::math::BabyBear>>>,
      16, 8>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_PLONKY3_CHALLENGER_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_

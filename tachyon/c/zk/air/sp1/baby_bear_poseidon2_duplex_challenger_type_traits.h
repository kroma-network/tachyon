#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger.h"
#include "tachyon/crypto/challenger/duplex_challenger.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::c::base {

namespace {

using Params = tachyon::crypto::Poseidon2Params<
    tachyon::math::BabyBear, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA>;

}  // namespace

template <>
struct TypeTraits<tachyon::crypto::DuplexChallenger<
    tachyon::crypto::Poseidon2Sponge<
        tachyon::crypto::Poseidon2ExternalMatrix<
            tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
                tachyon::math::BabyBear>>,
        Params>,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_duplex_challenger;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_duplex_challenger> {
  using NativeType = tachyon::crypto::DuplexChallenger<
      tachyon::crypto::Poseidon2Sponge<
          tachyon::crypto::Poseidon2ExternalMatrix<
              tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
                  tachyon::math::BabyBear>>,
          Params>,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_DUPLEX_CHALLENGER_TYPE_TRAITS_H_

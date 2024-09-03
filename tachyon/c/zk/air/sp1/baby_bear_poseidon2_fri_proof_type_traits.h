#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri_type_traits.h"
#include "tachyon/crypto/commitments/fri/fri_proof.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<
    tachyon::crypto::FRIProof<zk::air::plonky3::baby_bear::PCS::Base>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_fri_proof;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_fri_proof> {
  using NativeType =
      tachyon::crypto::FRIProof<zk::air::plonky3::baby_bear::PCS::Base>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FRI_PROOF_TYPE_TRAITS_H_

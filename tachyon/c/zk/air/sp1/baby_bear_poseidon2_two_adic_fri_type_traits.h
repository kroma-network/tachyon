#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/crypto/commitments/fri/two_adic_fri_impl.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_two_adic_fri.h"
#include "tachyon/crypto/challenger/duplex_challenger.h"
#include "tachyon/crypto/commitments/fri/two_adic_multiplicative_coset.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/extension_field_merkle_tree_mmcs.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"

namespace tachyon::c {
namespace zk::air::plonky3::baby_bear {

using Poseidon2 =
    tachyon::crypto::Poseidon2Sponge<tachyon::crypto::Poseidon2ExternalMatrix<
        tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
            tachyon::math::BabyBear>>>;

using PackedPoseidon2 =
    tachyon::crypto::Poseidon2Sponge<tachyon::crypto::Poseidon2ExternalMatrix<
        tachyon::crypto::Poseidon2Plonky3ExternalMatrix<
            tachyon::math::PackedBabyBear>>>;

using Hasher = tachyon::crypto::PaddingFreeSponge<
    Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using PackedHasher = tachyon::crypto::PaddingFreeSponge<
    PackedPoseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using Compressor = tachyon::crypto::TruncatedPermutation<
    Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_N>;

using PackedCompressor = tachyon::crypto::TruncatedPermutation<
    PackedPoseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_N>;

using Tree =
    tachyon::crypto::FieldMerkleTree<tachyon::math::BabyBear,
                                     TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using MMCS = tachyon::crypto::FieldMerkleTreeMMCS<
    tachyon::math::BabyBear, Hasher, PackedHasher, Compressor, PackedCompressor,
    TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using ExtMMCS = tachyon::crypto::FieldMerkleTreeMMCS<
    tachyon::math::BabyBear4, Hasher, PackedHasher, Compressor,
    PackedCompressor, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using ExtMMCS = tachyon::crypto::FieldMerkleTreeMMCS<
    tachyon::math::BabyBear4, Hasher, PackedHasher, Compressor,
    PackedCompressor, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

using ChallengeMMCS =
    tachyon::crypto::ExtensionFieldMerkleTreeMMCS<tachyon::math::BabyBear4,
                                                  ExtMMCS>;

using Challenger =
    tachyon::crypto::DuplexChallenger<Poseidon2,
                                      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
                                      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>;

using Coset =
    tachyon::crypto::TwoAdicMultiplicativeCoset<tachyon::math::BabyBear>;

using PCS = crypto::TwoAdicFRIImpl<tachyon::math::BabyBear4, MMCS,
                                   ChallengeMMCS, Challenger>;

}  // namespace zk::air::plonky3::baby_bear

namespace base {

template <>
struct TypeTraits<zk::air::plonky3::baby_bear::PCS> {
  using CType = tachyon_sp1_baby_bear_poseidon2_two_adic_fri;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_two_adic_fri> {
  using NativeType = zk::air::plonky3::baby_bear::PCS;
};

}  // namespace base
}  // namespace tachyon::c

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_TWO_ADIC_FRI_TYPE_TRAITS_H_

#ifndef TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_TYPE_TRAITS_H_
#define TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_TYPE_TRAITS_H_

#include <memory>
#include <vector>

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_constants.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::c {
namespace zk::air::plonky3::baby_bear {

using Tree =
    tachyon::crypto::FieldMerkleTree<tachyon::math::BabyBear,
                                     TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>;

}  // namespace zk::air::plonky3::baby_bear

namespace base {

template <>
struct TypeTraits<
    std::vector<std::unique_ptr<zk::air::plonky3::baby_bear::Tree>>> {
  using CType = tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec;
};

template <>
struct TypeTraits<tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec> {
  using NativeType =
      std::vector<std::unique_ptr<zk::air::plonky3::baby_bear::Tree>>;
};

}  // namespace base
}  // namespace tachyon::c

#endif  // TACHYON_C_ZK_AIR_SP1_BABY_BEAR_POSEIDON2_FIELD_MERKLE_TREE_VEC_TYPE_TRAITS_H_

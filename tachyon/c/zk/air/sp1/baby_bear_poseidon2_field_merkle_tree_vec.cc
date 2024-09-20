#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"

#include <vector>

#include "absl/debugging/leak_check.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec_type_traits.h"

using namespace tachyon;

using Tree = c::zk::air::sp1::baby_bear::Tree;
using TreeVec = std::vector<const Tree*>;

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create(size_t rounds) {
  return c::base::c_cast(new TreeVec(rounds));
}

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec) {
  const TreeVec& native_tree_vec = c::base::native_cast(*tree_vec);
  TreeVec* cloned = new TreeVec();
  *cloned = base::Map(native_tree_vec, [](const Tree* tree) {
    // NOTE(chokobole): The caller is responsible for deallocating the memory.
    // Specifically, in SP1 proof generation, the deallocation is handled by
    // calling |tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_destroy()|.
    // This function will be invoked by the Rust struct that owns the Field
    // Merkle tree.
    return absl::IgnoreLeak(new const Tree(*tree));
  });
  return c::base::c_cast(cloned);
}

void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec) {
  delete c::base::native_cast(tree_vec);
}

void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_set(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec,
    size_t round,
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree) {
  c::base::native_cast(*tree_vec)[round] = c::base::native_cast(tree);
}

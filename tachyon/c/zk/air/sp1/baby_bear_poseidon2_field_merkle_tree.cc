#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"

using namespace tachyon;

using Tree = c::zk::air::sp1::baby_bear::Tree;

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree) {
  return c::base::c_cast(new Tree(c::base::native_cast(*tree)));
}

void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree) {
  delete c::base::native_cast(tree);
}

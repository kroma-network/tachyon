#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec.h"

#include <memory>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_vec_type_traits.h"

using namespace tachyon;

using Tree = c::zk::air::plonky3::baby_bear::Tree;
using TreeVec = std::vector<std::unique_ptr<Tree>>;

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_create() {
  return c::base::c_cast(new TreeVec());
}

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_clone(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec) {
  const TreeVec& native_tree_vec = c::base::native_cast(*tree_vec);
  TreeVec* cloned = new TreeVec();
  *cloned = base::Map(native_tree_vec, [](const std::unique_ptr<Tree>& tree) {
    return std::make_unique<Tree>(*tree);
  });
  return c::base::c_cast(cloned);
}

void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_vec* tree_vec) {
  delete c::base::native_cast(tree_vec);
}

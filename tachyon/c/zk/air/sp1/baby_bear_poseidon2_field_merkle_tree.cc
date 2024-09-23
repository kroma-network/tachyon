#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree.h"

#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/buffer.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_field_merkle_tree_type_traits.h"

using namespace tachyon;

using F = math::BabyBear;
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

void tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_serialize(
    const tachyon_sp1_baby_bear_poseidon2_field_merkle_tree* tree,
    uint8_t* data, size_t* data_len) {
  *data_len = base::EstimateSize(c::base::native_cast(*tree));
  if (data == nullptr) return;

  base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                   true);
  base::Buffer buffer(data, *data_len);
  CHECK(buffer.Write(c::base::native_cast(*tree)));
  CHECK(buffer.Done());
}

tachyon_sp1_baby_bear_poseidon2_field_merkle_tree*
tachyon_sp1_baby_bear_poseidon2_field_merkle_tree_deserialize(
    const uint8_t* data, size_t data_len) {
  base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                   true);
  Tree* tree = new Tree();
  base::ReadOnlyBuffer buffer(data, data_len);
  CHECK(buffer.Read(tree));
  return c::base::c_cast(tree);
}

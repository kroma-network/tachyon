#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"

#include <vector>

#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/buffer.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values_type_traits.h"

using namespace tachyon;

using OpenedValues =
    std::vector<std::vector<std::vector<std::vector<math::BabyBear4>>>>;

tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_create() {
  return c::base::c_cast(new OpenedValues());
}

tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_clone(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values) {
  return c::base::c_cast(
      new OpenedValues(c::base::native_cast(*opened_values)));
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values) {
  delete c::base::native_cast(opened_values);
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values,
    uint8_t* data, size_t* data_len) {
  if (data == nullptr) {
    *data_len = base::EstimateSize(c::base::native_cast(*opened_values));
    return;
  }
  base::AutoReset<bool> auto_reset(
      &base::Copyable<math::BabyBear>::s_is_in_montgomery, true);
  base::Buffer buffer(data, *data_len);
  CHECK(buffer.Write(c::base::native_cast(*opened_values)));
}

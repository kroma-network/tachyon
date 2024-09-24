#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"

#include <vector>

#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/buffer.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_hintable.h"

using namespace tachyon;

using F = math::BabyBear;
using PCS = c::zk::air::sp1::baby_bear::PCS::Base;
using Proof = crypto::FRIProof<c::zk::air::sp1::baby_bear::PCS::Base>;

tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_create() {
  return c::base::c_cast(new Proof());
}

tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_clone(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof) {
  return c::base::c_cast(new Proof(c::base::native_cast(*fri_proof)));
}

void tachyon_sp1_baby_bear_poseidon2_fri_proof_destroy(
    tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof) {
  delete c::base::native_cast(fri_proof);
}

void tachyon_sp1_baby_bear_poseidon2_fri_proof_write_hint(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof, uint8_t* data,
    size_t* data_len) {
  std::vector<std::vector<c::zk::air::sp1::Block<F>>> hints =
      c::zk::air::sp1::baby_bear::WriteHint(c::base::native_cast(*fri_proof));
  *data_len = base::EstimateSize(hints);
  if (data == nullptr) return;

  base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                   true);
  base::Buffer buffer(data, *data_len);
  CHECK(buffer.Write(hints));
  CHECK(buffer.Done());
}

void tachyon_sp1_baby_bear_poseidon2_fri_proof_serialize(
    const tachyon_sp1_baby_bear_poseidon2_fri_proof* fri_proof, uint8_t* data,
    size_t* data_len) {
  *data_len = base::EstimateSize(c::base::native_cast(*fri_proof));
  if (data == nullptr) return;

  base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                   true);
  base::Buffer buffer(data, *data_len);
  CHECK(buffer.Write(c::base::native_cast(*fri_proof)));
  CHECK(buffer.Done());
}

tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_deserialize(const uint8_t* data,
                                                      size_t data_len) {
  base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                   true);
  Proof* fri_proof = new Proof();
  base::ReadOnlyBuffer buffer(data, data_len);
  CHECK(buffer.Read(fri_proof));
  return c::base::c_cast(fri_proof);
}

tachyon_sp1_baby_bear_poseidon2_fri_proof*
tachyon_sp1_baby_bear_poseidon2_fri_proof_deserialize_json(const uint8_t* data,
                                                           size_t data_len) {
  crypto::SP1FRIProof<PCS> sp1_proof;
  std::string error;
  CHECK(base::ParseJson(
      std::string_view(reinterpret_cast<const char*>(data), data_len),
      &sp1_proof, &error));
  CHECK(error.empty());
  Proof* proof = new Proof();
  *proof = std::move(sp1_proof.fri_proof);
  return c::base::c_cast(proof);
}

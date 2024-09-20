#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_fri_proof_type_traits.h"

using namespace tachyon;

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

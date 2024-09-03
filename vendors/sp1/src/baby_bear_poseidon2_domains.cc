#include "vendors/sp1/include/baby_bear_poseidon2_domains.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

Domains::~Domains() {
  tachyon_sp1_baby_bear_poseidon2_domains_destroy(domains_);
}

void Domains::allocate(size_t round, size_t size) {
  tachyon_sp1_baby_bear_poseidon2_domains_allocate(domains_, round, size);
}

void Domains::set(size_t round, size_t idx, uint32_t log_n,
                  const TachyonBabyBear& shift) {
  tachyon_sp1_baby_bear_poseidon2_domains_set(
      domains_, round, idx, log_n,
      reinterpret_cast<const tachyon_baby_bear*>(&shift));
}

std::unique_ptr<Domains> new_domains(size_t rounds) {
  return std::make_unique<Domains>(
      tachyon_sp1_baby_bear_poseidon2_domains_create(rounds));
}

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

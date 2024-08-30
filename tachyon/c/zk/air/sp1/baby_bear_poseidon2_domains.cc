#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains.h"

#include <vector>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains_type_traits.h"

using namespace tachyon;

using Domains = std::vector<
    std::vector<crypto::TwoAdicMultiplicativeCoset<math::BabyBear>>>;

tachyon_sp1_baby_bear_poseidon2_domains*
tachyon_sp1_baby_bear_poseidon2_domains_create(size_t rounds) {
  return c::base::c_cast(new Domains(rounds));
}

void tachyon_sp1_baby_bear_poseidon2_domains_destroy(
    tachyon_sp1_baby_bear_poseidon2_domains* domains) {
  delete c::base::native_cast(domains);
}

void tachyon_sp1_baby_bear_poseidon2_domains_allocate(
    tachyon_sp1_baby_bear_poseidon2_domains* domains, size_t round,
    size_t size) {
  c::base::native_cast(*domains)[round].resize(size);
}

void tachyon_sp1_baby_bear_poseidon2_domains_set(
    tachyon_sp1_baby_bear_poseidon2_domains* domains, size_t round, size_t idx,
    uint32_t log_n, const tachyon_baby_bear* shift) {
  c::base::native_cast(*domains)[round][idx] =
      crypto::TwoAdicMultiplicativeCoset(log_n, c::base::native_cast(*shift));
}

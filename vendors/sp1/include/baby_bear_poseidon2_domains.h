#ifndef VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_DOMAINS_H_
#define VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_DOMAINS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_domains.h"

namespace tachyon::sp1_api::baby_bear_poseidon2 {

struct TachyonBabyBear;

class Domains {
 public:
  explicit Domains(tachyon_sp1_baby_bear_poseidon2_domains* domains)
      : domains_(domains) {}
  Domains(const Domains& other) = delete;
  Domains& operator=(const Domains& other) = delete;
  ~Domains();

  const tachyon_sp1_baby_bear_poseidon2_domains* domains() const {
    return domains_;
  }

  void allocate(size_t round, size_t size);
  void set(size_t round, size_t idx, uint32_t log_n,
           const TachyonBabyBear& shift);

 private:
  tachyon_sp1_baby_bear_poseidon2_domains* domains_;
};

std::unique_ptr<Domains> new_domains(size_t rounds);

}  // namespace tachyon::sp1_api::baby_bear_poseidon2

#endif  // VENDORS_SP1_INCLUDE_BABY_BEAR_POSEIDON2_DOMAINS_H_

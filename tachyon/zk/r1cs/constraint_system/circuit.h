#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CIRCUIT_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CIRCUIT_H_

#include "tachyon/zk/r1cs/constraint_system/constraint_system.h"

namespace tachyon::zk::r1cs {

template <typename F>
class Circuit {
 public:
  virtual ~Circuit() = default;

  virtual void Synthesize(ConstraintSystem<F>& constraint_system) const = 0;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_CIRCUIT_H_

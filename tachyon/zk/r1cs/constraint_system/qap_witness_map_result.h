#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_WITNESS_MAP_RESULT_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_WITNESS_MAP_RESULT_H_

#include <memory_resource>
#include <vector>

namespace tachyon::zk::r1cs {

template <typename F>
struct QAPWitnessMapResult {
  std::pmr::vector<F> h;
  std::vector<F> full_assignments;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_WITNESS_MAP_RESULT_H_

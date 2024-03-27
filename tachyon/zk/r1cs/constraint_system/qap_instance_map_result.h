#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_INSTANCE_MAP_RESULT_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_INSTANCE_MAP_RESULT_H_

#include <stddef.h>

#include <vector>

namespace tachyon::zk::r1cs {

template <typename F>
struct QAPInstanceMapResult {
  std::vector<F> a;
  std::vector<F> b;
  std::vector<F> c;
  // t(x) = xⁿ⁺ˡ⁺¹ - 1
  F t_x;
  // n
  size_t num_constraints;
  // l + 1
  size_t num_instance_variables;
  // m - l
  size_t num_witness_variables;
  // m
  size_t num_qap_variables;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_QAP_INSTANCE_MAP_RESULT_H_

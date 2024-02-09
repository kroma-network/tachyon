#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TEST_SIMPLE_CIRCUIT_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TEST_SIMPLE_CIRCUIT_H_

#include <vector>

#include "tachyon/zk/r1cs/constraint_system/circuit.h"

namespace tachyon::zk::r1cs {

template <typename F>
class SimpleCircuit : public Circuit<F> {
 public:
  SimpleCircuit(const F& a, const F& b) : a_(a), b_(b) {}

  const F& a() const { return a_; }
  const F& b() const { return b_; }

  std::vector<F> GetPublicInputs() const { return {a_ * b_}; }

  void Synthesize(ConstraintSystem<F>& constraint_system) const override {
    Variable a =
        constraint_system.CreateWitnessVariable([this]() { return this->a(); });
    Variable b =
        constraint_system.CreateWitnessVariable([this]() { return this->b(); });
    Variable c = constraint_system.CreateInstanceVariable(
        [this]() { return this->a() * this->b(); });

    constraint_system.EnforceConstraint(LinearCombination<F>({{F::One(), a}}),
                                        LinearCombination<F>({{F::One(), b}}),
                                        LinearCombination<F>({{F::One(), c}}));
  }

 private:
  F a_;
  F b_;
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TEST_SIMPLE_CIRCUIT_H_

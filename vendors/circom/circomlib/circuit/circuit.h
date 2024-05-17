#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_H_

#include <string_view>
#include <vector>

#include "circomlib/circuit/witness_loader.h"
#include "circomlib/r1cs/r1cs.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/r1cs/constraint_system/circuit.h"

namespace tachyon::circom {

template <typename F>
class Circuit : public zk::r1cs::Circuit<F> {
 public:
  Circuit(R1CS<F>* r1cs, const base::FilePath& data)
      : r1cs_(r1cs), witness_loader_(data) {}

  WitnessLoader<F>& witness_loader() { return witness_loader_; }

  void Synthesize(
      zk::r1cs::ConstraintSystem<F>& constraint_system) const override {
    size_t i = 1;
    for (; i < r1cs_->GetNumInstanceVariables(); ++i) {
      constraint_system.CreateInstanceVariable(
          [this, i]() { return witness_loader_.Get(i); });
    }
    for (; i < r1cs_->GetNumVariables(); ++i) {
      constraint_system.CreateWitnessVariable(
          [this, i]() { return witness_loader_.Get(i); });
    }
    for (const Constraint<F>& constraint : r1cs_->GetConstraints()) {
      constraint_system.EnforceConstraint(
          MakeLC(constraint.a), MakeLC(constraint.b), MakeLC(constraint.c));
    }
  }

  std::vector<F> GetPublicInputs() const {
    return base::CreateVector(
        r1cs_->GetNumInstanceVariables() - 1,
        [this](size_t i) { return witness_loader_.Get(i + 1); });
  }

 private:
  zk::r1cs::LinearCombination<F> MakeLC(const LinearCombination<F>& lc) const {
    return zk::r1cs::LinearCombination<F>::CreateDeduplicated(
        base::Map(lc.terms, [this](const Term<F>& term) {
          zk::r1cs::Variable variable;
          if (term.wire_id < r1cs_->GetNumInstanceVariables()) {
            variable = zk::r1cs::Variable::Instance(term.wire_id);
          } else {
            variable = zk::r1cs::Variable::Witness(
                term.wire_id - r1cs_->GetNumInstanceVariables());
          }
          return zk::r1cs::Term<F>(term.coefficient, variable);
        }));
  }

  R1CS<F>* const r1cs_;
  WitnessLoader<F> witness_loader_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_CIRCUIT_H_

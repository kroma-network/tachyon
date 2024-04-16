// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TERM_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TERM_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/zk/r1cs/constraint_system/variable.h"

namespace tachyon::zk::r1cs {

template <typename F>
struct Term {
  F coefficient;
  Variable variable;

  constexpr Term() = default;
  constexpr explicit Term(const Variable& variable)
      : coefficient(F::One()), variable(variable) {}
  constexpr explicit Term(Variable&& variable)
      : coefficient(F::One()), variable(std::move(variable)) {}
  constexpr Term(const F& coefficient, const Variable& variable)
      : coefficient(coefficient), variable(variable) {}
  constexpr Term(F&& coefficient, const Variable& variable)
      : coefficient(std::move(coefficient)), variable(variable) {}
  constexpr Term(F&& coefficient, Variable&& variable)
      : coefficient(std::move(coefficient)), variable(std::move(variable)) {}

  Term operator-() const { return {-coefficient, variable}; }
  Term& NegateInPlace() {
    coefficient.NegateInPlace();
    return *this;
  }

  Term operator*(const F& scalar) const {
    return {this->coefficient * scalar, variable};
  }
  Term& operator*=(const F& scalar) {
    this->coefficient *= scalar;
    return *this;
  }

  bool operator==(const Term& other) const {
    return coefficient == other.coefficient && variable == other.variable;
  }
  bool operator!=(const Term& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("{coefficient: $0, variable: $1}",
                            coefficient.ToString(), variable.ToString());
  }
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_SYSTEM_TERM_H_

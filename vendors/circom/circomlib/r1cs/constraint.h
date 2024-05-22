#ifndef VENDORS_CIRCOM_CIRCOMLIB_R1CS_CONSTRAINT_H_
#define VENDORS_CIRCOM_CIRCOMLIB_R1CS_CONSTRAINT_H_

#include <stdint.h>

#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"

namespace tachyon::circom {

template <typename F>
struct Term {
  uint32_t wire_id;
  F coefficient;

  bool operator==(const Term& other) const {
    return wire_id == other.wire_id && coefficient == other.coefficient;
  }
  bool operator!=(const Term& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("$0ω_$1", coefficient.ToString(), wire_id);
  }
};

template <typename F>
struct LinearCombination {
  std::vector<Term<F>> terms;

  bool operator==(const LinearCombination& other) const {
    return terms == other.terms;
  }
  bool operator!=(const LinearCombination& other) const {
    return terms != other.terms;
  }

  std::string ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < terms.size(); ++i) {
      ss << terms[i].ToString();
      if (i != terms.size() - 1) ss << " + ";
    }
    return ss.str();
  }
};

template <typename F>
struct Constraint {
  LinearCombination<F> a;
  LinearCombination<F> b;
  LinearCombination<F> c;

  bool operator==(const Constraint& other) const {
    return a == other.a && b == other.b && c == other.c;
  }
  bool operator!=(const Constraint& other) const { return !operator==(other); }

  std::string ToString() const {
    if (a.terms.size() > 1 && b.terms.size() > 1) {
      return absl::Substitute("($0) * ($1) = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else if (a.terms.size() > 1) {
      return absl::Substitute("($0) * $1 = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else if (b.terms.size() > 1) {
      return absl::Substitute("$0 * ($1) = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else {
      return absl::Substitute("$0 * $1 = $2", a.ToString(), b.ToString(),
                              c.ToString());
    }
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_R1CS_CONSTRAINT_H_

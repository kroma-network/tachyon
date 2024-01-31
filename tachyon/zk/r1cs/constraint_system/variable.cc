// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#include "tachyon/zk/r1cs/constraint_system/variable.h"

#include "absl/strings/substitute.h"

namespace tachyon::zk::r1cs {

// static
std::string_view Variable::TypeToString(Variable::Type type) {
  switch (type) {
    case Variable::Type::kZero:
      return "Zero";
    case Variable::Type::kOne:
      return "One";
    case Variable::Type::kInstance:
      return "Instance";
    case Variable::Type::kWitness:
      return "Witness";
    case Variable::Type::kSymbolicLinearCombination:
      return "SymbolicLinearCombination";
  }
  NOTREACHED();
  return "";
}

std::string Variable::ToString() const {
  if (type_ == Variable::Type::kZero || type_ == Variable::Type::kOne) {
    return absl::Substitute("{type: $0}", TypeToString(type_));
  }
  return absl::Substitute("{type: $0, index: $1}", TypeToString(type_), index_);
}

}  // namespace tachyon::zk::r1cs

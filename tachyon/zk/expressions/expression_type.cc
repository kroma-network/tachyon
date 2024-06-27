// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/expressions/expression_type.h"

#include "tachyon/base/logging.h"

namespace tachyon::zk {

std::string_view ExpressionTypeToString(ExpressionType type) {
  switch (type) {
    case ExpressionType::kConstant:
      return "Constant";
    case ExpressionType::kNegated:
      return "Negated";
    case ExpressionType::kSum:
      return "Sum";
    case ExpressionType::kProduct:
      return "Product";
    case ExpressionType::kScaled:
      return "Scaled";
    case ExpressionType::kSelector:
      return "Selector";
    case ExpressionType::kFixed:
      return "Fixed";
    case ExpressionType::kAdvice:
      return "Advice";
    case ExpressionType::kInstance:
      return "Instance";
    case ExpressionType::kChallenge:
      return "Challenge";
  }
  NOTREACHED();
  return "";
}

std::ostream& operator<<(std::ostream& os, ExpressionType type) {
  return os << ExpressionTypeToString(type);
}

}  // namespace tachyon::zk

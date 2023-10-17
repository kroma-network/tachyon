// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_TYPE_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_TYPE_H_

#include <ostream>
#include <string_view>

#include "tachyon/export.h"

namespace tachyon::zk {

enum class ExpressionType {
  kConstant,
  kSelector,
  kFixed,
  kAdvice,
  kInstance,
  kChallenge,
  kNegated,
  kSum,
  kProduct,
  kScaled
};

TACHYON_EXPORT std::string_view ExpressionTypeToString(ExpressionType type);

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, ExpressionType type);

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_TYPE_H_

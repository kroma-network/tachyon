// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/vanishing/calculation.h"

#include "absl/strings/substitute.h"

#include "tachyon/base/strings/string_util.h"

namespace tachyon::zk {

std::string Calculation::ToString() const {
  switch (type_) {
    case Type::kAdd:
      return absl::Substitute("Add($0, $1)", pair_.left.ToString(),
                              pair_.right.ToString());
    case Type::kSub:
      return absl::Substitute("Sub($0, $1)", pair_.left.ToString(),
                              pair_.right.ToString());
    case Type::kMul:
      return absl::Substitute("Mul($0, $1)", pair_.left.ToString(),
                              pair_.right.ToString());
    case Type::kSquare:
      return absl::Substitute("Square($0)", value_.ToString());
    case Type::kDouble:
      return absl::Substitute("Double($0)", value_.ToString());
    case Type::kNegate:
      return absl::Substitute("Negate($0)", value_.ToString());
    case Type::kStore:
      return absl::Substitute("Store($0)", value_.ToString());
    case Type::kHorner:
      return absl::Substitute("Horner($0, $1, $2)", horner_.init.ToString(),
                              horner_.factor.ToString(),
                              base::VectorToString(horner_.parts));
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk

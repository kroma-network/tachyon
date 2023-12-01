// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/vanishing/value_source.h"

#include "absl/strings/substitute.h"

namespace tachyon::zk {

std::string ValueSource::ToString() const {
  switch (type_) {
    case Type::kConstant:
      return absl::Substitute("Constant($0)", index_);
    case Type::kIntermediate:
      return absl::Substitute("Intermediate($0)", index_);
    case Type::kFixed:
      return absl::Substitute("Fixed($0, $1)", column_index_, rotation_index_);
    case Type::kAdvice:
      return absl::Substitute("Advice($0, $1)", column_index_, rotation_index_);
    case Type::kInstance:
      return absl::Substitute("Instance($0, $1)", column_index_,
                              rotation_index_);
    case Type::kChallenge:
      return absl::Substitute("Challenge($0)", index_);
    case Type::kBeta:
      return "Beta";
    case Type::kGamma:
      return "Gamma";
    case Type::kTheta:
      return "Theta";
    case Type::kY:
      return "Y";
    case Type::kPreviousValue:
      return "PreviousValue";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk

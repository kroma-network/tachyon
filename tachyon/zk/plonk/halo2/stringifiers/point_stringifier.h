// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_POINT_STRINGIFIER_H_
#define TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_POINT_STRINGIFIER_H_

#include <ostream>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/zk/plonk/halo2/stringifiers/field_stringifier.h"

namespace tachyon::base::internal {

template <typename Curve>
class RustDebugStringifier<math::AffinePoint<Curve>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const math::AffinePoint<Curve>& value) {
    if (value.IsZero()) {
      return os << "Infinity";
    } else {
      return os
             << fmt.DebugTuple("").Field(value.x()).Field(value.y()).Finish();
    }
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_PLONK_HALO2_STRINGIFIERS_POINT_STRINGIFIER_H_

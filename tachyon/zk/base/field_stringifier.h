// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_FIELD_STRINGIFIER_H_
#define TACHYON_ZK_BASE_FIELD_STRINGIFIER_H_

#include <ostream>
#include <type_traits>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::base::internal {

template <typename T>
class RustDebugStringifier<
    T, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<T>, T>>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const T& value) {
    return os << value.ToHexString(true);
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_BASE_FIELD_STRINGIFIER_H_

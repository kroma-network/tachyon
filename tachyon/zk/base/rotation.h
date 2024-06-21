// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ROTATION_H_
#define TACHYON_ZK_BASE_ROTATION_H_

#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/zk/base/row_types.h"

namespace tachyon::zk {

// NOTE(lightscale-luke): Rotation class is copyable, assignable, and occupies
// 32 bits per instance. Prefer to pass them by value.
class TACHYON_EXPORT Rotation {
 public:
  Rotation() = default;
  explicit Rotation(int32_t value) : value_(value) {}

  static Rotation Cur() { return Rotation(0); }

  static Rotation Prev() { return Rotation(-1); }

  static Rotation Next() { return Rotation(1); }

  int32_t value() const { return value_; }

  bool operator==(const Rotation& other) const {
    return value_ == other.value_;
  }
  bool operator!=(const Rotation& other) const {
    return value_ != other.value_;
  }

  std::string ToString() const { return base::NumberToString(value_); }

  // Returns (|idx| + |value_| * |scale|) modulo |size|.
  RowIndex GetIndex(int32_t idx, int32_t scale, int32_t size) const {
    CHECK_GT(size, 0);
    base::CheckedNumeric<int32_t> value = value_;
    int32_t result = ((idx + value * scale) % size).ValueOrDie();
    if (result < 0) result += size;
    return result;
  }

  template <typename Domain, typename F>
  F RotateOmega(const Domain* domain, const F& point) const {
    return point * domain->GetElement(value_);
  }

 private:
  int32_t value_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ROTATION_H_

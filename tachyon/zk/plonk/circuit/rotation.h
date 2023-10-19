// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_ROTATION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_ROTATION_H_

#include <stdint.h>

#include <string>

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_number_conversions.h"

namespace tachyon::zk {

// NOTE(lightscale-luke): Rotation class is copyable, assignable, and occupy 64
// bits per instance. Prefer to pass them by value.
class TACHYON_EXPORT Rotation {
 public:
  Rotation() = default;
  explicit Rotation(int32_t value) : value_(value) {}

  static Rotation Cur() { return Rotation(0); }

  static Rotation Prev() { return Rotation(-1); }

  static Rotation Next() { return Rotation(1); }

  int32_t value() const { return value_; }

  std::string ToString() const { return base::NumberToString(value_); }

  // Returns ((|idx| + |value_| * |scale|) % |size|).
  // It fails when |idx| + |value_| * |scale| evaluates to be negative.
  size_t GetIndex(int32_t idx, int32_t scale, int32_t size) const {
    int32_t value = idx + value_ * scale;
    CHECK_GE(value, int32_t{0});
    return size_t{base::bit_cast<uint32_t>(value % size)};
  }

 private:
  int32_t value_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_ROTATION_H_

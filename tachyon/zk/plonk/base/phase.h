// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_BASE_PHASE_H_
#define TACHYON_ZK_PLONK_BASE_PHASE_H_

#include <stdint.h>

#include <string>
#include <utility>

#include "absl/hash/hash.h"

#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/export.h"

namespace tachyon::zk {

// NOTE(lightscale-luke): Phase class is copyable, assignable, and occupy 8
// bits per instance. Prefer to pass them by value.
class TACHYON_EXPORT Phase {
 public:
  constexpr Phase() = default;
  constexpr explicit Phase(uint8_t value) : value_(value) {}

  bool Prev(Phase* prev) const {
    if (value_ == 0) return false;
    *prev = Phase(value_ - 1);
    return true;
  }

  uint8_t value() const { return value_; }

  bool operator==(Phase other) const { return value_ == other.value_; }
  bool operator!=(Phase other) const { return value_ != other.value_; }
  bool operator>(Phase other) const { return value_ > other.value_; }
  bool operator>=(Phase other) const { return value_ >= other.value_; }
  bool operator<(Phase other) const { return value_ < other.value_; }
  bool operator<=(Phase other) const { return value_ <= other.value_; }

  std::string ToString() const { return base::NumberToString(value_); }

 private:
  uint8_t value_ = 0;
};

constexpr static Phase kFirstPhase = Phase(0);
constexpr static Phase kSecondPhase = Phase(1);

template <typename H>
H AbslHashValue(H h, Phase phase) {
  return H::combine(std::move(h), phase.value());
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_BASE_PHASE_H_

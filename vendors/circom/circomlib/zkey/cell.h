#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CELL_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CELL_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/zk/r1cs/constraint_system/matrix.h"

namespace tachyon::circom {

template <typename F>
struct Cell {
  F coefficient;
  uint32_t signal;

  bool operator==(const Cell& other) const {
    return coefficient == other.coefficient && signal == other.signal;
  }
  bool operator!=(const Cell& other) const { return !operator==(other); }

  // Need to divide by R, since snarkjs outputs the zkey with coefficients
  // multiplied by R².
  zk::r1cs::Cell<F> ToNative() const {
    return {
        F::FromMontgomery(coefficient.ToBigInt()),
        signal,
    };
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute("($0, $1)", coefficient.ToString(), signal);
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CELL_H_

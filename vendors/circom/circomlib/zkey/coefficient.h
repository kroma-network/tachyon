#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_COEFFICIENT_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_COEFFICIENT_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

namespace tachyon::circom {

#pragma pack(push, 1)
// R1CS is represented as A * z ∘ B * z = C * z, where ∘ is the Hadamard
// product. Each constraint is composed as follows:
//
// - [aᵢ,₀, ...,  aᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
// - [bᵢ,₀, ...,  bᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
// - [cᵢ,₀, ...,  cᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
//
// where i is the index of the constraints (0 ≤ i < n),
// m is the number of QAP variables, and n is the number of constraints.
//
// The last constraint is computed if we know the first two constraints.
// Therefore, |Coefficient| represents the first two constraints.
template <typename F>
struct Coefficient {
  // A value denoting the matrix this constraint is for. If 0, this constraint
  // is for matrix A. Else, this constraint is for matrix B.
  uint32_t matrix;
  // The index of the constraint, (0 ≤ i < n).
  uint32_t constraint;
  // The index of the QAP variables, (0 ≤ j < m).
  uint32_t signal;
  // The values of the coefficient; if the |matrix| is 0, then this points to
  // the a[i][j]. Otherwise, this points to b[i][j].
  F value;

  bool operator==(const Coefficient& other) const {
    return matrix == other.matrix && constraint == other.constraint &&
           signal == other.signal && value == other.value;
  }
  bool operator!=(const Coefficient& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute(
        "{matrix: $0, constraint: $1, signal: $2, value: $3}", matrix,
        constraint, signal, value.ToString());
  }
};
#pragma pack(pop)

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_COEFFICIENT_H_

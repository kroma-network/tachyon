// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_CONSTRAINT_MATRICES_H_
#define TACHYON_ZK_R1CS_CONSTRAINT_MATRICES_H_

#include <stddef.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/zk/r1cs/matrix.h"

namespace tachyon::zk::r1cs {

// The A, B and C matrices of a Rank-One |ConstraintSystem|.
// Also contains metadata on the structure of the constraint system
// and the matrices.
// FIXME(chokobole): I want to separate solution vectors from A, B and C
// matrices since the solution vectors are shared. This is because we can save
// additional memory space if we adopt this trick.
template <typename F>
struct ConstraintMatrices {
  // The number of variables that are "public instances" to the constraint
  // system.
  size_t num_instance_variables = 0;
  // The number of variables that are "private witnesses" to the constraint
  // system.
  size_t num_witness_variables = 0;
  // The number of constraints in the constraint system.
  size_t num_constraints = 0;
  // The number of non_zero entries in the A matrix.
  size_t a_num_non_zero = 0;
  // The number of non_zero entries in the B matrix.
  size_t b_num_non_zero = 0;
  // The number of non_zero entries in the C matrix.
  size_t c_num_non_zero = 0;

  // The A constraint matrix.
  Matrix<F> a;
  // The B constraint matrix.
  Matrix<F> b;
  // The C constraint matrix.
  Matrix<F> c;

  bool operator==(const ConstraintMatrices& other) const {
    return num_instance_variables == other.num_instance_variables &&
           num_witness_variables == other.num_witness_variables &&
           num_constraints == other.num_constraints &&
           a_num_non_zero == other.a_num_non_zero &&
           b_num_non_zero == other.b_num_non_zero &&
           c_num_non_zero == other.c_num_non_zero && a == other.a &&
           b == other.b && c == other.c;
  }
  bool operator!=(const ConstraintMatrices& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute(
        "{ num_instance_variables: $0, num_witness_variables: $1, "
        "num_constraints: $2, a_num_non_zero: $3, b_num_non_zero: $4, "
        "c_num_non_zero: $5, a: $6, b: $7, c: $8",
        num_instance_variables, num_witness_variables, num_constraints,
        a_num_non_zero, b_num_non_zero, c_num_non_zero, a.ToString(),
        b.ToString(), c.ToString());
  }
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_CONSTRAINT_MATRICES_H_

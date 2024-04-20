#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CONSTRAINT_MATRICES_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CONSTRAINT_MATRICES_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/substitute.h"

#include "circomlib/zkey/cell.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/zk/r1cs/constraint_system/constraint_matrices.h"

namespace tachyon::circom {

struct ConstraintMatrices {
  size_t num_instance_variables;
  size_t num_witness_variables;
  size_t num_constraints;

  size_t a_num_non_zero;
  size_t b_num_non_zero;

  std::vector<std::vector<Cell>> a;
  std::vector<std::vector<Cell>> b;

  bool operator==(const ConstraintMatrices& other) const {
    return num_instance_variables == other.num_instance_variables &&
           num_witness_variables == other.num_witness_variables &&
           num_constraints == other.num_constraints &&
           a_num_non_zero == other.a_num_non_zero &&
           b_num_non_zero == other.b_num_non_zero && a == other.a &&
           b == other.b;
  }
  bool operator!=(const ConstraintMatrices& other) const {
    return !operator==(other);
  }

  template <typename F>
  zk::r1cs::ConstraintMatrices<F> ToNative() const {
    return {
        num_instance_variables,
        num_witness_variables,
        num_constraints,
        a_num_non_zero,
        b_num_non_zero,
        0,
        zk::r1cs::Matrix<F>(base::Map(a,
                                      [](const std::vector<Cell>& cells) {
                                        return base::Map(
                                            cells, [](const Cell& cell) {
                                              return cell.ToNative<F>();
                                            });
                                      })),
        zk::r1cs::Matrix<F>(base::Map(b,
                                      [](const std::vector<Cell>& cells) {
                                        return base::Map(
                                            cells, [](const Cell& cell) {
                                              return cell.ToNative<F>();
                                            });
                                      })),
        {},
    };
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const {
    return absl::Substitute(
        "{num_instance_variables: $0, num_witness_variables: $1, "
        "num_constraints: $2, a_num_non_zero: $3, b_num_non_zero: $4, a: $5, "
        "b: $6}",
        num_instance_variables, num_witness_variables, num_constraints,
        a_num_non_zero, b_num_non_zero, base::Container2DToString(a),
        base::Container2DToString(b));
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_CONSTRAINT_MATRICES_H_

// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_EXTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_EXTERNAL_MATRIX_H_

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_external_matrix.h"

namespace tachyon::crypto {

template <typename F>
class Poseidon2Plonky3ExternalMatrix final
    : public Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>> {
 public:
  template <typename Derived>
  static void DoApply(Eigen::MatrixBase<Derived>& v) {
    F t0 = v(0, 0) + v(1, 0);
    F t1 = v(2, 0) + v(3, 0);
    F t2 = t0 + t1;
    F t3 = t2 + v(1, 0);
    F t4 = t2 + v(3, 0);
    v(3, 0) = t4 + v(0, 0).Double();
    v(1, 0) = t3 + v(2, 0).Double();
    v(0, 0) = t3 + t0;
    v(2, 0) = t4 + t1;
  }

  static math::Matrix<F> DoConstruct() {
    return math::Matrix<F>{
        {F(2), F(3), F(1), F(1)},
        {F(1), F(2), F(3), F(1)},
        {F(1), F(1), F(2), F(3)},
        {F(3), F(1), F(1), F(2)},
    };
  }
};

template <typename Field_>
struct Poseidon2ExternalMatrixTraits<Poseidon2Plonky3ExternalMatrix<Field_>> {
  using Field = Field_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_EXTERNAL_MATRIX_H_

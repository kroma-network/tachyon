// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_HORIZEN_EXTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_HORIZEN_EXTERNAL_MATRIX_H_

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_external_matrix.h"

namespace tachyon::crypto {

template <typename PrimeField>
class Poseidon2HorizenExternalMatrix final
    : public Poseidon2ExternalMatrix<
          Poseidon2HorizenExternalMatrix<PrimeField>> {
 public:
  static void DoApply(math::Vector<PrimeField>& v) {
    PrimeField t0 = v[0] + v[1];
    PrimeField t1 = v[2] + v[3];
    PrimeField t2 = v[1] + v[1] + t1;
    PrimeField t3 = v[3] + v[3] + t0;
    v[3] = t1.Double().Double() + t3;
    v[1] = t0.Double().Double() + t2;
    v[0] = t3 + v[1];
    v[2] = t2 + v[3];
  }

  static math::Matrix<PrimeField> DoConstruct() {
    return math::Matrix<PrimeField>{
        {PrimeField(5), PrimeField(7), PrimeField(1), PrimeField(3)},
        {PrimeField(4), PrimeField(6), PrimeField(1), PrimeField(1)},
        {PrimeField(1), PrimeField(3), PrimeField(5), PrimeField(7)},
        {PrimeField(1), PrimeField(1), PrimeField(4), PrimeField(6)},
    };
  }
};

template <typename PrimeField_>
struct Poseidon2ExternalMatrixTraits<
    Poseidon2HorizenExternalMatrix<PrimeField_>> {
  using PrimeField = PrimeField_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_HORIZEN_EXTERNAL_MATRIX_H_

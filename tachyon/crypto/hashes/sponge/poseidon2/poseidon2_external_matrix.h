// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_

#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_external_matrix_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon::crypto {

template <typename Derived>
class Poseidon2ExternalMatrix {
 public:
  using Field = typename Poseidon2ExternalMatrixTraits<Derived>::Field;

  static void Apply(math::Vector<Field>& v) {
    size_t size = v.size();
    if (size <= 1 || size > 24) {
      NOTREACHED() << "Out of range";
    } else if (size == 2) {
      Field sum = v[0] + v[1];
      v[0] += sum;
      v[1] += sum;
      return;
    } else if (size == 3) {
      Field sum = v[0] + v[1] + v[2];
      v[0] += sum;
      v[1] += sum;
      v[2] += sum;
      return;
    }

    if (size % 4 != 0) {
      NOTREACHED() << "Not a multiple of 4";
    }

    if (size == 4) {
      Derived::DoApply(v);
      return;
    }

    math::Vector<Field> v_tmp(4);
    for (size_t i = 0; i < size; i += 4) {
      v_tmp << v[i], v[i + 1], v[i + 2], v[i + 3];
      Derived::DoApply(v_tmp);
      v.block(i, 0, 4, 1) = v_tmp;
    }

    v_tmp << Field::Zero(), Field::Zero(), Field::Zero(), Field::Zero();
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < size; j += 4) {
        v_tmp[i] += v[i + j];
      }
    }

    for (size_t i = 0; i < size; ++i) {
      v[i] += v_tmp[i % 4];
    }
  }

  static math::Matrix<Field> Construct(size_t size) {
    if (size <= 1 || size > 24) {
      NOTREACHED() << "Out of range";
    } else if (size == 2) {
      return math::MakeCirculant(math::Vector<Field>{{Field(2), Field(1)}});
    } else if (size == 3) {
      return math::MakeCirculant(
          math::Vector<Field>{{Field(2), Field(1), Field(1)}});
    }
    if (size % 4 != 0) {
      NOTREACHED() << "Not a multiple of 4";
    }

    math::Matrix<Field> small_matrix = Derived::DoConstruct();

    if (size == 4) {
      return small_matrix;
    } else {
      math::Matrix<Field> ret(size, size);
      for (size_t i = 0; i < size / 4; ++i) {
        for (size_t j = 0; j < size / 4; ++j) {
          ret.block(i * 4, j * 4, 4, 4) =
              i == j ? Field(2) * small_matrix : small_matrix;
        }
      }
      return ret;
    }
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_

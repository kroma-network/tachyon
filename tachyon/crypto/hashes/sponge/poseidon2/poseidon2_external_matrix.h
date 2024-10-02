// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_

#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_external_matrix_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon::crypto {

template <typename Derived>
class Poseidon2ExternalMatrix {
 public:
  using Field = typename Poseidon2ExternalMatrixTraits<Derived>::Field;

  template <size_t N>
  static void Apply(math::Vector<Field>& v) {
    if constexpr (N <= 1 || N > 24) {
      NOTREACHED() << "Out of range";
    } else if constexpr (N == 2) {
      Field sum = v[0] + v[1];
      v[0] += sum;
      v[1] += sum;
    } else if constexpr (N == 3) {
      Field sum = v[0] + v[1] + v[2];
      v[0] += sum;
      v[1] += sum;
      v[2] += sum;
    } else if constexpr (N == 4) {
      Derived::DoApply(v);
    } else if constexpr (N % 4 == 0) {
      for (size_t i = 0; i < N; i += 4) {
        Eigen::Block<math::Vector<Field>> block = v.block(i, 0, 4, 1);
        Derived::DoApply(block);
      }

      std::array<Field, 4> v_tmp = {Field::Zero()};
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < N; j += 4) {
          v_tmp[i] += v[i + j];
        }
      }

      for (size_t i = 0; i < N; ++i) {
        v[i] += v_tmp[i % 4];
      }
    } else {
      NOTREACHED() << "Not a multiple of 4";
    }
  }

  template <size_t N>
  static math::Matrix<Field> Construct() {
    if constexpr (N <= 1 || N > 24) {
      NOTREACHED() << "Out of range";
    } else if constexpr (N == 2) {
      return math::MakeCirculant(math::Vector<Field>{{Field(2), Field(1)}});
    } else if constexpr (N == 3) {
      return math::MakeCirculant(
          math::Vector<Field>{{Field(2), Field(1), Field(1)}});
    } else if constexpr (N == 4) {
      return Derived::DoConstruct();
    } else if constexpr (N % 4 == 0) {
      math::Matrix<Field> small_matrix = Derived::DoConstruct();

      math::Matrix<Field> ret(N, N);
      for (size_t i = 0; i < N / 4; ++i) {
        for (size_t j = 0; j < N / 4; ++j) {
          ret.block(i * 4, j * 4, 4, 4) =
              i == j ? Field(2) * small_matrix : small_matrix;
        }
      }
      return ret;
    } else {
      NOTREACHED() << "Not a multiple of 4";
    }
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_EXTERNAL_MATRIX_H_

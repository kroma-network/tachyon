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

  template <size_t kWidth>
  static void Apply(math::Vector<Field>& v) {
    if constexpr (kWidth <= 1 || kWidth > 24) {
      NOTREACHED() << "Out of range";
    } else if constexpr (kWidth == 2) {
      Field sum = v[0] + v[1];
      v[0] += sum;
      v[1] += sum;
    } else if constexpr (kWidth == 3) {
      Field sum = v[0] + v[1] + v[2];
      v[0] += sum;
      v[1] += sum;
      v[2] += sum;
    } else if constexpr (kWidth == 4) {
      Derived::DoApply(v);
    } else if constexpr (kWidth % 4 == 0) {
      for (size_t i = 0; i < kWidth; i += 4) {
        Eigen::Block<math::Vector<Field>> block = v.block(i, 0, 4, 1);
        Derived::DoApply(block);
      }

      std::array<Field, 4> v_tmp = {Field::Zero(), Field::Zero(), Field::Zero(),
                                    Field::Zero()};
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < kWidth; j += 4) {
          v_tmp[i] += v[i + j];
        }
      }

      for (size_t i = 0; i < kWidth; ++i) {
        v[i] += v_tmp[i % 4];
      }
    } else {
      NOTREACHED() << "Not a multiple of 4";
    }
  }

  template <size_t kWidth>
  static math::Matrix<Field> Construct() {
    if constexpr (kWidth <= 1 || kWidth > 24) {
      NOTREACHED() << "Out of range";
    } else if constexpr (kWidth == 2) {
      return math::MakeCirculant(math::Vector<Field>{{Field(2), Field(1)}});
    } else if constexpr (kWidth == 3) {
      return math::MakeCirculant(
          math::Vector<Field>{{Field(2), Field(1), Field(1)}});
    } else if constexpr (kWidth == 4) {
      return Derived::DoConstruct();
    } else if constexpr (kWidth % 4 == 0) {
      math::Matrix<Field> small_matrix = Derived::DoConstruct();

      math::Matrix<Field> ret(kWidth, kWidth);
      for (size_t i = 0; i < kWidth / 4; ++i) {
        for (size_t j = 0; j < kWidth / 4; ++j) {
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

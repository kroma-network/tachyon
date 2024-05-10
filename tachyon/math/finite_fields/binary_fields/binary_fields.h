// Copyright 2023 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#ifndef TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELDS_H_
#define TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELDS_H_

// clang-format off
#include "tachyon/math/finite_fields/binary_fields/binary_field1_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field2_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field4_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field8_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field16_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field32_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field64_config.h"
// clang-format on
#include "tachyon/math/finite_fields/binary_field_traits_forward.h"

namespace tachyon::math {

template <>
struct BinaryFieldTraits<BinaryField1Config> {
  using SubConfig = BinaryField1Config;
};

template <>
struct BinaryFieldTraits<BinaryField2Config> {
  using SubConfig = BinaryField2Config;
};

template <>
struct BinaryFieldTraits<BinaryField4Config> {
  using SubConfig = BinaryField4Config;
};

template <>
struct BinaryFieldTraits<BinaryField8Config> {
  using SubConfig = BinaryField8Config;
};

template <>
struct BinaryFieldTraits<BinaryField16Config> {
  using SubConfig = BinaryField8Config;
};

template <>
struct BinaryFieldTraits<BinaryField32Config> {
  using SubConfig = BinaryField16Config;
};

template <>
struct BinaryFieldTraits<BinaryField64Config> {
  using SubConfig = BinaryField32Config;
};

}  // namespace tachyon::math

// clang-format off
#include "tachyon/math/finite_fields/binary_fields/binary_field1.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field2.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field4.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field8.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field16.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field32.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field64.h"
// clang-format on

#endif  // TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELDS_H_

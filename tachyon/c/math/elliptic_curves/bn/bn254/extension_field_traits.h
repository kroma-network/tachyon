#ifndef TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_EXTENSION_FIELD_TRAITS_H_
#define TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_EXTENSION_FIELD_TRAITS_H_

#include "tachyon/c/math/elliptic_curves/bn/bn254/fp12.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fp6.h"
#include "tachyon/cc/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq12.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq2.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq6.h"

namespace tachyon::cc::math {

template <>
struct ExtensionFieldTraits<tachyon::math::bn254::Fq2> {
  using CExtensionField = tachyon_bn254_fp2;
};

template <>
struct ExtensionFieldTraits<tachyon::math::bn254::Fq6> {
  using CExtensionField = tachyon_bn254_fp6;
};

template <>
struct ExtensionFieldTraits<tachyon::math::bn254::Fq12> {
  using CExtensionField = tachyon_bn254_fp12;
};

template <>
struct ExtensionFieldTraits<tachyon_bn254_fp2> {
  using ExtensionField = tachyon::math::bn254::Fq2;
};

template <>
struct ExtensionFieldTraits<tachyon_bn254_fp6> {
  using ExtensionField = tachyon::math::bn254::Fq6;
};

template <>
struct ExtensionFieldTraits<tachyon_bn254_fp12> {
  using ExtensionField = tachyon::math::bn254::Fq12;
};

}  // namespace tachyon::cc::math

#endif  // TACHYON_C_MATH_ELLIPTIC_CURVES_BN_BN254_EXTENSION_FIELD_TRAITS_H_

#ifndef TACHYON_CC_MATH_FINITE_FIELDS_EXTENSION_FIELD_CONVERSIONS_H_
#define TACHYON_CC_MATH_FINITE_FIELDS_EXTENSION_FIELD_CONVERSIONS_H_

#include <stddef.h>

#include "tachyon/cc/math/finite_fields/extension_field_traits_forward.h"

namespace tachyon::cc::math {

template <typename CExtensionField,
          typename ExtensionField =
              typename ExtensionFieldTraits<CExtensionField>::ExtensionField>
const ExtensionField& native_cast(const CExtensionField& f) {
  static_assert(sizeof(ExtensionField) == sizeof(CExtensionField));
  return reinterpret_cast<const ExtensionField&>(f);
}

template <typename CExtensionField,
          typename ExtensionField =
              typename ExtensionFieldTraits<CExtensionField>::ExtensionField>
ExtensionField& native_cast(CExtensionField& f) {
  static_assert(sizeof(ExtensionField) == sizeof(CExtensionField));
  return reinterpret_cast<ExtensionField&>(f);
}

template <typename ExtensionField,
          typename CExtensionField =
              typename ExtensionFieldTraits<ExtensionField>::CExtensionField>
const CExtensionField& c_cast(const ExtensionField& f) {
  static_assert(sizeof(CExtensionField) == sizeof(ExtensionField));
  return reinterpret_cast<const CExtensionField&>(f);
}

template <typename ExtensionField,
          typename CExtensionField =
              typename ExtensionFieldTraits<ExtensionField>::CExtensionField>
CExtensionField& c_cast(ExtensionField& f) {
  static_assert(sizeof(CExtensionField) == sizeof(ExtensionField));
  return reinterpret_cast<CExtensionField&>(f);
}

}  // namespace tachyon::cc::math

#endif  // TACHYON_CC_MATH_FINITE_FIELDS_EXTENSION_FIELD_CONVERSIONS_H_

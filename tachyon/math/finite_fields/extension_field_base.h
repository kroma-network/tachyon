// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_EXTENSION_FIELD_BASE_H_
#define TACHYON_MATH_FINITE_FIELDS_EXTENSION_FIELD_BASE_H_

#include <iterator>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/extended_packed_field_traits_forward.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/packed_field_traits_forward.h"

namespace tachyon::math {

template <typename Derived>
class ExtensionFieldBase {
 public:
  class ConstIterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = typename ExtensionFieldTraits<Derived>::BaseField;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    ConstIterator(const Derived& derived, size_t idx)
        : derived_(derived), idx_(idx) {}

    bool operator==(const ConstIterator& other) const {
      return &derived_ == &other.derived_ && idx_ == other.idx_;
    }
    bool operator!=(const ConstIterator& other) const {
      return !(*this == other);
    }

    ConstIterator& operator++() {
      ++idx_;
      return *this;
    }

    ConstIterator operator++(int) {
      ConstIterator iterator(*this);
      ++(*this);
      return iterator;
    }

    const pointer operator->() const { return &derived_[idx_]; }

    const value_type& operator*() const { return derived_[idx_]; }

   private:
    const Derived& derived_;
    size_t idx_;
  };

  template <
      typename T,
      typename ExtendedPackedField =
          typename ExtendedPackedFieldTraits<T>::ExtendedPackedField,
      std::enable_if_t<!ExtendedPackedFieldTraits<T>::kIsExtendedPackedField>* =
          nullptr>
  static std::vector<ExtendedPackedField> GetExtendedPackedPowers(const T& base,
                                                                  size_t size) {
    using BaseField = typename T::BaseField;
    using PackedField = typename PackedFieldTraits<BaseField>::PackedField;
    uint32_t degree = T::ExtensionDegree();
    T pow = T::One();
    std::array<std::array<BaseField, T::ExtensionDegree()>, PackedField::N + 1>
        powers_base_field =
            base::CreateArray<PackedField::N + 1>([&base, &pow]() {
              auto ret = pow.ToBaseFields();
              pow *= base;
              return ret;
            });

    // Transpose first WIDTH powers
    ExtendedPackedField first_n_powers;
    // Broadcast self^WIDTH
    ExtendedPackedField multiplier;
    for (uint32_t deg = 0; deg < degree; ++deg) {
      first_n_powers[deg] =
          PackedField::From([&powers_base_field, deg](size_t j) {
            return powers_base_field[j][deg];
          });
      multiplier[deg] = PackedField::From([&powers_base_field, deg](size_t j) {
        return powers_base_field[PackedField::N][deg];
      });
    }

    std::vector<ExtendedPackedField> ret;
    ret.reserve(size);
    ret.emplace_back(first_n_powers);
    for (size_t i = 0; i < size - 1; ++i) {
      ret.emplace_back(ret[i] * multiplier);
    }
    return ret;
  }

  ConstIterator begin() const {
    return {static_cast<const Derived&>(*this), 0};
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_EXTENSION_FIELD_BASE_H_

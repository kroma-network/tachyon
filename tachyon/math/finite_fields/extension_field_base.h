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
    constexpr uint32_t kDegree = T::ExtensionDegree();

    // if |PackedField::N| = 8,
    // |first_n_powers| = [ 1, b, b², b³,..., b⁶, b⁷ ], where b is an extension
    // field of |base|.
    ExtendedPackedField first_n_powers;
    T pow = T::One();
    for (size_t i = 0; i < PackedField::N; ++i) {
      for (uint32_t j = 0; j < kDegree; ++j) {
        first_n_powers[j][i] = pow[j];
      }
      pow *= base;
    }

    // |multiplier| = [ b⁸, b⁸, ..., b⁸, b⁸ ], where b is an extension field of
    // |base|. #|multiplier| = 8
    ExtendedPackedField multiplier;
    for (size_t i = 0; i < PackedField::N; ++i) {
      for (uint32_t j = 0; j < kDegree; ++j) {
        multiplier[j][i] = pow[j];
      }
    }

    // |ret[i]| = [ b⁸ⁱ, b⁸ⁱ⁺¹, ..., b⁸ⁱ⁺⁶, b⁸ⁱ⁺⁷ ], where b is an extension
    // field of |base|.
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

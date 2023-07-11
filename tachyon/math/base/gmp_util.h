#ifndef TACHYON_MATH_BASE_GMP_UTIL_H_
#define TACHYON_MATH_BASE_GMP_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <iterator>
#include <limits>
#include <string_view>
#include <type_traits>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/build/build_config.h"
#include "tachyon/export.h"

namespace tachyon {
namespace math {
namespace gmp {

class TACHYON_EXPORT BitIteratorBE {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = bool;
  using pointer = const bool*;
  using reference = const bool&;
  using iterator_category = std::forward_iterator_tag;

  constexpr explicit BitIteratorBE(const mpz_class* field)
      : BitIteratorBE(field, 0) {}
  constexpr BitIteratorBE(const mpz_class* field, size_t index)
      : field_(field), index_(index) {}
  constexpr BitIteratorBE(const BitIteratorBE& other) = default;
  constexpr BitIteratorBE& operator=(const BitIteratorBE& other) = default;

  static BitIteratorBE begin(const mpz_class* field);
  static BitIteratorBE end(const mpz_class* field);

  constexpr bool operator==(const BitIteratorBE& other) const {
    return field_ == other.field_ && index_ == other.index_;
  }
  constexpr bool operator!=(const BitIteratorBE& other) const {
    return !(*this == other);
  }

  constexpr BitIteratorBE& operator++() {
#if ARCH_CPU_BIG_ENDIAN == 1
    ++index_;
#else
    --index_;
#endif
    return *this;
  }

  constexpr BitIteratorBE operator++(int) {
    BitIteratorBE it(*this);
    ++(*this);
    return it;
  }

  const bool* operator->() const;

  const bool& operator*() const;

 private:
  const mpz_class* field_ = nullptr;
  size_t index_ = 0;
  mutable bool value_ = false;
};

class TACHYON_EXPORT BitIteratorLE {
 public:
  using difference_type = std::ptrdiff_t;
  using value_type = bool;
  using pointer = const bool*;
  using reference = const bool&;
  using iterator_category = std::forward_iterator_tag;

  constexpr explicit BitIteratorLE(const mpz_class* field)
      : BitIteratorLE(field, 0) {}
  constexpr BitIteratorLE(const mpz_class* field, size_t index)
      : field_(field), index_(index) {}
  constexpr BitIteratorLE(const BitIteratorLE& other) = default;
  constexpr BitIteratorLE& operator=(const BitIteratorLE& other) = default;

  static BitIteratorLE begin(const mpz_class* field);
  static BitIteratorLE end(const mpz_class* field);

  bool operator==(const BitIteratorLE& other) const {
    return field_ == other.field_ && index_ == other.index_;
  }
  bool operator!=(const BitIteratorLE& other) const {
    return !(*this == other);
  }

  constexpr BitIteratorLE& operator++() {
#if ARCH_CPU_LITTLE_ENDIAN == 1
    ++index_;
#else
    --index_;
#endif
    return *this;
  }

  constexpr BitIteratorLE operator++(int) {
    BitIteratorLE it(*this);
    ++(*this);
    return it;
  }

  const bool* operator->() const;

  const bool& operator*() const;

 private:
  constexpr BitIteratorLE& operator--() {
#if ARCH_CPU_LITTLE_ENDIAN == 1
    --index_;
#else
    ++index_;
#endif
    return *this;
  }

  constexpr BitIteratorLE operator--(int) {
    BitIteratorLE it(*this);
    --(*this);
    return it;
  }

  const mpz_class* field_ = nullptr;
  size_t index_ = 0;
  mutable bool value_ = false;
};

TACHYON_EXPORT gmp_randstate_t& GetRandomState();

TACHYON_EXPORT bool ParseIntoMpz(std::string_view str, int base,
                                 mpz_class* out);

TACHYON_EXPORT void MustParseIntoMpz(std::string_view str, int base,
                                     mpz_class* out);

TACHYON_EXPORT void UnsignedIntegerToMpz(unsigned long int value,
                                         mpz_class* out);

TACHYON_EXPORT bool IsZero(const mpz_class& out);
TACHYON_EXPORT bool IsNegative(const mpz_class& out);
TACHYON_EXPORT bool IsPositive(const mpz_class& out);

TACHYON_EXPORT size_t GetNumBits(const mpz_class& value);
TACHYON_EXPORT size_t GetLimbSize(const mpz_class& value);
TACHYON_EXPORT uint64_t GetLimb(const mpz_class& value, size_t idx);

}  // namespace gmp
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_GMP_UTIL_H_

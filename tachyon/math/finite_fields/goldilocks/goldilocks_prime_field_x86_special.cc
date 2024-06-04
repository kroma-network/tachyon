#if defined(TACHYON_HAS_ASM_PRIME_FIELD)

#include "tachyon/math/finite_fields/goldilocks/goldilocks_prime_field_x86_special.h"

#include <optional>

#include "third_party/goldilocks/include/goldilocks_base_field.hpp"

#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/invalid_operation.h"

namespace tachyon::math {

#define CLASS \
  PrimeField<Config, std::enable_if_t<Config::kIsTachyonMathGoldilocks>>

template <typename Config>
CLASS::PrimeField(uint64_t value) : value_(value) {
  DCHECK_LT(value_, Config::kModulus[0]);
  value_ = ::Goldilocks::fromU64(value_).fe;
}

// static
template <typename Config>
CLASS CLASS::One() {
  return PrimeField(::Goldilocks::one().fe);
}

// static
template <typename Config>
CLASS CLASS::Random() {
  return PrimeField(
      ::Goldilocks::fromU64(
          base::Uniform(base::Range<uint64_t>::Until(Config::kModulus[0])))
          .fe);
}

// static
template <typename Config>
std::optional<CLASS> CLASS::FromDecString(std::string_view str) {
  uint64_t value = 0;
  if (!base::StringToUint64(str, &value)) return std::nullopt;
  if (value >= Config::kModulus[0]) {
    LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
    return std::nullopt;
  }
  return PrimeField(value);
}

// static
template <typename Config>
std::optional<CLASS> CLASS::FromHexString(std::string_view str) {
  uint64_t value = 0;
  if (!base::HexStringToUint64(str, &value)) return std::nullopt;
  if (value >= Config::kModulus[0]) {
    LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
    return std::nullopt;
  }
  return PrimeField(value);
}

// static
template <typename Config>
CLASS CLASS::FromBigInt(BigInt<N> big_int) {
  return PrimeField(big_int[0]);
}

#if USE_MONTGOMERY == 1
// static
template <typename Config>
CLASS CLASS::FromMontgomery(BigInt<N> big_int) {
  PrimeField ret;
  ret.value_ = big_int[0];
  return ret;
}
#endif

template <typename Config>
bool CLASS::IsZero() const {
  return ::Goldilocks::isZero(
      reinterpret_cast<const ::Goldilocks::Element&>(value_));
}

template <typename Config>
bool CLASS::IsOne() const {
  return ::Goldilocks::isOne(
      reinterpret_cast<const ::Goldilocks::Element&>(value_));
}

template <typename Config>
std::string CLASS::ToString() const {
  return ::Goldilocks::toString(
      reinterpret_cast<const ::Goldilocks::Element&>(value_), 10);
}

template <typename Config>
std::string CLASS::ToHexString(bool pad_zero) const {
  std::string str = ::Goldilocks::toString(
      reinterpret_cast<const ::Goldilocks::Element&>(value_), 16);
  if (pad_zero) {
    str = base::ToHexStringWithLeadingZero(str, 16);
  }
  return base::MaybePrepend0x(str);
}

template <typename Config>
CLASS CLASS::Add(const PrimeField& other) const {
  PrimeField ret;
  ::Goldilocks::add(
      reinterpret_cast<::Goldilocks::Element&>(ret.value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::AddInPlace(const PrimeField& other) {
  ::Goldilocks::add(
      reinterpret_cast<::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return *this;
}

template <typename Config>
CLASS CLASS::Sub(const PrimeField& other) const {
  PrimeField ret;
  ::Goldilocks::sub(
      reinterpret_cast<::Goldilocks::Element&>(ret.value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::SubInPlace(const PrimeField& other) {
  ::Goldilocks::sub(
      reinterpret_cast<::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return *this;
}

template <typename Config>
CLASS CLASS::Negate() const {
  PrimeField ret;
  ::Goldilocks::neg(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::NegateInPlace() {
  ::Goldilocks::neg(reinterpret_cast<::Goldilocks::Element&>(value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return *this;
}

template <typename Config>
CLASS CLASS::Mul(const PrimeField& other) const {
  PrimeField ret;
  ::Goldilocks::mul(
      reinterpret_cast<::Goldilocks::Element&>(ret.value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::MulInPlace(const PrimeField& other) {
  ::Goldilocks::mul(
      reinterpret_cast<::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(value_),
      reinterpret_cast<const ::Goldilocks::Element&>(other.value_));
  return *this;
}

template <typename Config>
CLASS CLASS::SquareImpl() const {
  PrimeField ret;
  ::Goldilocks::square(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                       reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::SquareImplInPlace() {
  ::Goldilocks::square(reinterpret_cast<::Goldilocks::Element&>(value_),
                       reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return *this;
}

template <typename Config>
std::optional<CLASS> CLASS::Inverse() const {
  if (UNLIKELY(InvalidOperation(IsZero(), "Inverse of zero attempted"))) {
    return std::nullopt;
  }
  PrimeField ret;
  ::Goldilocks::inv(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
std::optional<CLASS*> CLASS::InverseInPlace() {
  if (UNLIKELY(InvalidOperation(IsZero(), "Inverse of zero attempted"))) {
    return std::nullopt;
  }
  ::Goldilocks::inv(reinterpret_cast<::Goldilocks::Element&>(value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return this;
}

#undef CLASS

template class PrimeField<GoldilocksConfig>;

}  // namespace tachyon::math

#endif  // TACHYON_HAS_ASM_PRIME_FIELD

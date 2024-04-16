#include "tachyon/math/finite_fields/goldilocks/goldilocks_prime_field_x86_special.h"

#include "third_party/goldilocks/include/goldilocks_base_field.hpp"

#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon::math {

#define CLASS PrimeField<Config, std::enable_if_t<Config::kIsGoldilocks>>

template <typename Config>
CLASS::PrimeField(uint64_t value) : value_(value) {}

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
  return PrimeField(::Goldilocks::fromU64(value).fe);
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
  return PrimeField(::Goldilocks::fromU64(value).fe);
}

// static
template <typename Config>
CLASS CLASS::FromBigInt(const BigInt<N>& big_int) {
  return PrimeField(::Goldilocks::fromU64(big_int[0]).fe);
}

// static
template <typename Config>
CLASS CLASS::FromMontgomery(const BigInt<N>& big_int) {
  return PrimeField(::Goldilocks::from_montgomery(big_int[0]));
}

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
mpz_class CLASS::ToMpzClass() const {
  mpz_class ret;
  uint64_t limbs[] = {uint64_t{*this}};
  gmp::WriteLimbs(limbs, N, &ret);
  return ret;
}

template <typename Config>
BigInt<CLASS::N> CLASS::ToMontgomery() const {
  return BigInt<N>(::Goldilocks::to_montgomery(uint64_t{*this}));
}

template <typename Config>
CLASS::operator uint64_t() const {
  return ::Goldilocks::toU64(::Goldilocks::Element{value_});
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
CLASS CLASS::Negative() const {
  PrimeField ret;
  ::Goldilocks::neg(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::NegInPlace() {
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
CLASS CLASS::DoSquare() const {
  PrimeField ret;
  ::Goldilocks::square(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                       reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::DoSquareInPlace() {
  ::Goldilocks::square(reinterpret_cast<::Goldilocks::Element&>(value_),
                       reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return *this;
}

template <typename Config>
CLASS CLASS::Inverse() const {
  // See https://github.com/kroma-network/tachyon/issues/76
  CHECK(!IsZero());
  PrimeField ret;
  ::Goldilocks::inv(reinterpret_cast<::Goldilocks::Element&>(ret.value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return ret;
}

template <typename Config>
CLASS& CLASS::InverseInPlace() {
  // See https://github.com/kroma-network/tachyon/issues/76
  CHECK(!IsZero());
  ::Goldilocks::inv(reinterpret_cast<::Goldilocks::Element&>(value_),
                    reinterpret_cast<const ::Goldilocks::Element&>(value_));
  return *this;
}

#undef CLASS

template class PrimeField<GoldilocksConfig>;

}  // namespace tachyon::math

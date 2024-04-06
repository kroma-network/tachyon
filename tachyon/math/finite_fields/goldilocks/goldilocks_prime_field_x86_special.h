#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks_config.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<_Config::kIsGoldilocks>> final
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = uint64_t;

  PrimeField() = default;
  explicit PrimeField(uint64_t value);
  PrimeField(const PrimeField& other) = default;
  PrimeField& operator=(const PrimeField& other) = default;
  PrimeField(PrimeField&& other) = default;
  PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }
  static PrimeField One();
  static PrimeField Random();

  static PrimeField FromDecString(std::string_view str);
  static PrimeField FromHexString(std::string_view str);
  static PrimeField FromBigInt(const BigInt<N>& big_int);
  static PrimeField FromMontgomery(const BigInt<N>& big_int);

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  const value_type& value() const { return value_; }

  bool IsZero() const;
  bool IsOne() const;

  std::string ToString() const;
  std::string ToHexString(bool pad_zero = false) const;

  mpz_class ToMpzClass() const;

  // TODO(chokobole): Support bigendian.
  BigInt<N> ToBigInt() const { return BigInt<N>(uint64_t{*this}); }

  BigInt<N> ToMontgomery() const;

  operator uint64_t() const;

  uint64_t operator[](size_t i) const {
    DCHECK_EQ(i, size_t{0});
    return uint64_t{*this};
  }

  bool operator==(const PrimeField& other) const {
    return uint64_t{*this} == uint64_t{other};
  }
  bool operator!=(const PrimeField& other) const {
    return uint64_t{*this} != uint64_t{other};
  }
  bool operator<(const PrimeField& other) const {
    return uint64_t{*this} < uint64_t{other};
  }
  bool operator>(const PrimeField& other) const {
    return uint64_t{*this} > uint64_t{other};
  }
  bool operator<=(const PrimeField& other) const {
    return uint64_t{*this} <= uint64_t{other};
  }
  bool operator>=(const PrimeField& other) const {
    return uint64_t{*this} >= uint64_t{other};
  }

  // AdditiveSemigroup methods
  PrimeField Add(const PrimeField& other) const;
  PrimeField& AddInPlace(const PrimeField& other);
  PrimeField& DoubleInPlace();

  // AdditiveGroup methods
  PrimeField& SubInPlace(const PrimeField& other);
  PrimeField& NegInPlace();

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  PrimeField Mul(const PrimeField& other) const;
  PrimeField& MulInPlace(const PrimeField& other);
  PrimeField& DivInPlace(const PrimeField& other);
  PrimeField& SquareInPlace();

  // MultiplicativeGroup methods
  PrimeField& InverseInPlace();

 private:
  uint64_t value_;
};

extern template class PrimeField<GoldilocksConfig>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_

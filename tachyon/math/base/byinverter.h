#ifndef TACHYON_MATH_BASE_BYINVERTER_H_
#define TACHYON_MATH_BASE_BYINVERTER_H_

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#include "absl/numeric/int128.h"

#include "tachyon/base/bits.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon::math {

// Unsigned 128-bit integer type
typedef absl::uint128 uint128_t;

// Big signed (B * L)-bit integer type, whose variables
// store numbers in the two's complement code as arrays
// of B-bit chunks, where B < 64 and L > 0. The ordering
// of the chunks in these arrays is little-endian. The
// arithmetic operations for this type are wrapping ones
template <size_t B, size_t L>
class CInt {
 public:
  // Mask, in which the B lowest bits are 1 and only they
  constexpr static uint64_t MASK = ~uint64_t{0} >> (64 - B);

  // Returns the number equal to the argument. The binary
  // representation of the absolute value of the argument
  // must have no more than B * L - 1 bits to ensure that
  // the result and the argument represent the same number
  constexpr static CInt Load(const int64_t input) {
    CInt result;
    uint64_t raw = input;
    Convert<64, B>(&raw, 1, result.data, L);
    if ((B * L > 64) && (input < 0)) {
      // For the two's complement code and n > m the n-bit representation
      // of a negative number can be expressed as its m-bit representation
      // preceded by m - n bits equal to 1
      result.data[64 / B] |= (~uint64_t{0} << (64 % B)) & MASK;
      for (size_t i = 64 / B + 1; i < L; i++) {
        result.data[i] = MASK;
      }
    }
    return result;
  }

  // Returns the number, the absolute value of which is specified by the
  // first argument. A non-zero return number is negative iff "sign" is
  // "true". The binary representation of the absolute value of the input
  // number must have no more than B * L - 1 bits to ensure that the result
  // and arguments specify the same number
  template <size_t N>
  constexpr static CInt Load(const BigInt<N> &input, bool sign) {
    CInt result;
    Convert<64, B>(input.limbs, N, result.data, L);
    return sign ? -result : result;
  }

  // Returns the current number. The binary representation
  // of its absolute value must have no more than 63 bits
  // to ensure that the result represents the current number
  constexpr int64_t Save() const {
    uint64_t result = 0;
    Convert<B, 64>(data, L, &result, 1);
    if ((B * L < 64) && IsNegative()) {
      // For the two's complement code and n > m the n-bit representation
      // of a negative number can be expressed as its m-bit representation
      // preceded by m - n bits equal to 1
      result |= ~uint64_t{0} << (B * L);
    }
    return result;
  }

  // Returns the sign of the current number and stores its absolute
  // value into the variable, which is specified by the argument.
  // The binary representation of the absolute value of the current
  // number must contain no more than 64 * N bits to ensure that the
  // result represents the current number
  template <size_t N>
  constexpr bool Save(BigInt<N> &output) const {
    bool sign = IsNegative();
    Convert<B, 64>((sign ? -*this : *this).data, L, output.limbs, N);
    return sign;
  }

  // Returns the lowest B bits of the current number
  constexpr uint64_t Lowest() const { return data[0]; }

  // Returns "true" iff the current number is negative
  constexpr bool IsNegative() const { return data[L - 1] > (MASK >> 1); }

  // Returns the result of applying B-bit right
  // arithmetical shift to the current number
  constexpr CInt Shift() const {
    CInt result;
    if (IsNegative()) {
      result.data[L - 1] = MASK;
    }
    for (size_t i = 1; i < L; i++) {
      result.data[i - 1] = data[i];
    }
    return result;
  }

  constexpr bool operator==(const CInt &other) const {
    for (size_t i = 0; i < L; i++) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }

  constexpr bool operator!=(const CInt &other) const {
    return !(*this == other);
  }

  constexpr CInt operator+(const CInt &other) const {
    CInt result;
    uint64_t carry = 0;
    for (size_t i = 0; i < L; i++) {
      uint64_t sum = data[i] + other.data[i] + carry;
      result.data[i] = sum & MASK;
      carry = sum >> B;
    }
    return result;
  }

  constexpr CInt operator-(const CInt &other) const {
    // For the two's complement code the additive negation is the result of
    // adding 1 to the bitwise inverted argument's representation. Thus, for
    // any encoded integers x and y we have x - y = x + !y + 1, where "!" is
    // the bitwise inversion and addition is done according to the rules of
    // the code. The algorithm below uses this formula and is the modified
    // addition algorithm, where the carry flag is initialized with 1 and
    // the chunks of the second argument are bitwise inverted
    CInt result;
    uint64_t carry = 1;
    for (size_t i = 0; i < L; i++) {
      uint64_t sum = data[i] + (other.data[i] ^ MASK) + carry;
      result.data[i] = sum & MASK;
      carry = sum >> B;
    }
    return result;
  }

  constexpr CInt operator-() const {
    // For the two's complement code the additive negation is the result
    // of adding 1 to the bitwise inverted argument's representation
    CInt result;
    uint64_t carry = 1;
    for (size_t i = 0; i < L; i++) {
      uint64_t sum = (data[i] ^ MASK) + carry;
      result.data[i] = sum & MASK;
      carry = sum >> B;
    }
    return result;
  }

  constexpr CInt operator*(const CInt &other) const {
    CInt result;
    for (size_t i = 0; i < L; i++) {
      uint64_t carry = 0;
      for (size_t k = i; k < L; k++) {
        uint128_t sum = static_cast<uint128_t>(data[i]) * other.data[k - i] +
                        result.data[k] + carry;
        result.data[k] = static_cast<uint64_t>(sum & MASK);
        carry = static_cast<uint64_t>(sum >> B);
      }
    }
    return result;
  }

  constexpr CInt operator*(int64_t other) const {
    CInt result;
    uint64_t mask = 0;
    uint64_t carry = 0;
    // If the short multiplicand is non-negative, the standard multiplication
    // algorithm is performed. Otherwise, the product of the additively negated
    // multiplicands is found as follows. Since for the two's complement code
    // the additive negation is the result of adding 1 to the bitwise inverted
    // argument's representation, for any encoded integers x and y we have
    // x * y = (-x) * (-y) = (!x + 1) * (-y) = !x * (-y) + (-y),  where "!" is
    // the bitwise inversion and arithmetic operations are performed according
    // to the rules of the code. If the short multiplicand is negative, the
    // algorithm below uses this formula by substituting the short multiplicand
    // for y and turns into the modified standard multiplication algorithm,
    // where the carry flag is initialized with the additively negated short
    // multiplicand and the chunks of the long multiplicand are bitwise inverted
    if (other < 0) {
      mask = MASK;
      carry = other = -other;
    }
    for (size_t i = 0; i < L; i++) {
      uint128_t sum = static_cast<uint128_t>(data[i] ^ mask) * other + carry;
      result.data[i] = static_cast<uint64_t>(sum & MASK);
      carry = static_cast<uint64_t>(sum >> B);
    }
    return result;
  }

 private:
  // B-bit chunks representing the current number
  uint64_t data[L] = {0};

  // Creates an instance representing zero
  constexpr CInt() {}

  // Stores at the address "output" the array of O-bit chunks, which
  // represents the big unsigned integer equal modulo 2ᴼ * "ᵒˢᶦᶻᵉ" to
  // the input big unsigned integer stored at the address "input" as
  // an array of "isize" I-bit chunks. The ordering of the chunks in
  // these arrays is little-endian
  template <size_t I, size_t O>
  constexpr static void Convert(const uint64_t *input, const size_t isize,
                                uint64_t *output, const size_t osize) {
    size_t bits = 0;
    size_t total = std::min(isize * I, osize * O);
    for (size_t i = 0; i < osize; i++) {
      output[i] = 0;
    }
    while (bits < total) {
      size_t i = bits % I;
      size_t o = bits % O;
      output[bits / O] |= (input[bits / I] >> i) << o;
      bits += std::min(I - i, O - o);
    }
    uint64_t mask = ~uint64_t{0} >> (64 - O);
    size_t filled = (total + O - 1) / O;
    for (size_t i = 0; i < filled; i++) {
      output[i] &= mask;
    }
  }
};

// Type of the modular multiplicative inverter based on the Bernstein-Yang
// method. The inverter can be created for a specified odd modulus M and
// adjusting parameter A to compute the adjusted multiplicative inverses
// of positive integers, i.e. for computing (1 / x) * A (mod M) for a
// positive integer x.
//
// The adjusting parameter allows computing the multiplicative inverses
// in the case of using the Montgomery representation for the input or
// the expected output. If R is the Montgomery factor, the multiplicative
// inverses in the appropriate representation can be computed provided
// that the value of A is chosen as follows:
// - A = 1, if both the input and expected output are in the trivial form;
// - A = R² mod M, if both the input and the expected output are in the
// Montgomery form;
// - A = R mod M, if either the input or the expected output is in the
// Montgomery form, but not both of them.
//
// For a better understanding of the implementation, use following resources:
// - D. Bernstein, B.-Y. Yang, "Fast constant-time gcd computation and modular
// inversion", https://gcd.cr.yp.to/safegcd-20190413.pdf
// - P. Wuille, "The safegcd implementation in libsecp256k1 explained",
// https://github.com/bitcoin-core/secp256k1/blob/master/doc/safegcd_implementation.md
template <size_t L>
class BYInverter {
 public:
  // Creates an inverter for specified modulus and adjusting parameter
  constexpr BYInverter(const BigInt<L> &modulus, const BigInt<L> &adjuster)
      : modulus(LCInt::Load(modulus, false)),
        adjuster(LCInt::Load(adjuster, false)),
        inverse62(Invert62(modulus.limbs[0])) {}

  // Returns "true" and stores into the variable, which is specified by
  // the second argument, the adjusted modular multiplicative inverse
  // of the input number, if it is invertible for the modulus of the
  // invertor (i.e. coprime with it). Otherwise, "false" is returned
  bool Invert(const BigInt<L> &input, BigInt<L> &output) const {
    int64_t delta = 1;
    LCInt f = modulus;
    LCInt g = LCInt::Load(input, false);
    LCInt d = ZERO;
    LCInt e = adjuster;
    int64_t t[2][2];
    while (g != ZERO) {
      Jump(f.Lowest(), g.Lowest(), delta, t);
      FG(f, g, t);
      DE(d, e, t);
    }
    // At this point the absolute value of "f" equals the greatest
    // common divisor of the integer to be inverted and the modulus
    // the inverter was created for. Thus, if "f" is neither 1 nor
    // -1, then the sought inverse does not exist
    bool antiunit = f == MINUSONE;
    bool invertible = (f == ONE) || antiunit;
    if (invertible) {
      Norm(d, antiunit).Save(output);
    }
    return invertible;
  }

 private:
  // The big signed integer type used by the inverter. The absolute
  // values of big signed results of the intermediate computations
  // have no more than 64 * L + 63 bits in their binary representation
  typedef CInt<62, (64 * (L + 1) + 61) / 62> LCInt;

  // The big signed representation of 0
  constexpr static LCInt ZERO = LCInt::Load(0);

  // The big signed representation of 1
  constexpr static LCInt ONE = LCInt::Load(1);

  // The big signed representation of -1
  constexpr static LCInt MINUSONE = LCInt::Load(-1);

  // Modulus
  LCInt modulus;

  // Adjusting parameter
  LCInt adjuster;

  // Multiplicative inverse of the modulus modulo 2⁶²
  uint64_t inverse62;

  // Stores the Bernstein-Yang transition matrix multiplied by 2⁶² and
  // the new value of the delta variable for the 62 basic steps of the
  // Bernstein-Yang method, which are to be performed sequentially for
  // specified initial values of delta, f and g. The initial values of
  // f and g are specified partially: only the least significant chunks
  // of their LCInt representations are the arguments
  static void Jump(uint64_t f, uint64_t g, int64_t &delta, int64_t t[2][2]) {
    t[0][0] = t[1][1] = 1;
    t[0][1] = t[1][0] = 0;
    int64_t steps = 62;
    int64_t y[2];
    uint64_t x;

    while (true) {
      int64_t zeros = base::bits::CountTrailingZeroBits(g);
      zeros = std::min(zeros, steps);
      steps -= zeros;
      delta += zeros;
      g >>= zeros;
      t[0][0] <<= zeros;
      t[0][1] <<= zeros;

      if (steps == 0) {
        break;
      }

      if (delta > 0) {
        delta = -delta;

        y[0] = -t[0][0];
        y[1] = -t[0][1];
        t[0][0] = t[1][0];
        t[0][1] = t[1][1];
        t[1][0] = y[0];
        t[1][1] = y[1];

        x = -f;
        f = g;
        g = x;
      }

      // The formula (3 * x) xor 28 = -1 / x (mod 32) for an odd integer
      // x in the two's complement code has been derived from the formula
      // (3 * x) xor 2 = 1 / x (mod 32) attributed to Peter Montgomery
      uint64_t mask =
          (1 << std::min(std::min(steps, 1 - delta), int64_t{5})) - 1;
      uint64_t w = g * ((3 * f) ^ 28) & mask;

      y[0] = t[0][0] * w + t[1][0];
      y[1] = t[0][1] * w + t[1][1];
      t[1][0] = y[0];
      t[1][1] = y[1];

      g += w * f;
    }
  }

  // Stores the updated values of the variables f and g for specified
  // initial ones and Bernstein-Yang transition matrix multiplied by 2⁶².
  // In the vector form this operation can be described using the formula:
  // "(f, g)' := matrix * (f, g)' / 2⁶²", where "'" is the transpose
  // operator and ":=" denotes assignment
  static void FG(LCInt &f, LCInt &g, const int64_t t[2][2]) {
    LCInt x = (f * t[0][0] + g * t[0][1]).Shift();
    LCInt y = (f * t[1][0] + g * t[1][1]).Shift();
    f = x;
    g = y;
  }

  // Stores the updated values of the variables d and e for specified
  // initial ones and Bernstein-Yang transition matrix multiplied by
  // 2⁶². The new value of the vector "(d, e)'" is congruent modulo M
  // to "matrix * (d, e)' / 2⁶² (mod M)", where M is the modulus the
  // inverter was created for and "'" stands for the transpose operator.
  // Both the initial and new values of d and e lie in (-2 * M, M)
  void DE(LCInt &d, LCInt &e, const int64_t (&t)[2][2]) const {
    int64_t md = t[0][0] * d.IsNegative() + t[0][1] * e.IsNegative();
    int64_t me = t[1][0] * d.IsNegative() + t[1][1] * e.IsNegative();
    {
      int64_t cd = (t[0][0] * d.Lowest() + t[0][1] * e.Lowest()) & LCInt::MASK;
      int64_t ce = (t[1][0] * d.Lowest() + t[1][1] * e.Lowest()) & LCInt::MASK;

      md -= (cd * inverse62 + md) & LCInt::MASK;
      me -= (ce * inverse62 + me) & LCInt::MASK;
    }
    LCInt cd = d * t[0][0] + e * t[0][1] + modulus * md;
    LCInt ce = d * t[1][0] + e * t[1][1] + modulus * me;
    d = cd.Shift();
    e = ce.Shift();
  }

  // Returns either "value (mod M)" or "-value (mod M)", where
  // M is the modulus the inverter was created for, depending
  // on "negate", which determines the presence of "-" in the
  // used formula. The input integer lies in (-2 * M, M)
  LCInt Norm(const LCInt &value, const bool negate) const {
    LCInt result = value.IsNegative() ? value + modulus : value;
    if (negate) {
      result = -result;
    }
    if (result.IsNegative()) {
      result = result + modulus;
    }
    return result;
  }

  // Returns the multiplicative inverse of the argument modulo 2⁶². The
  // implementation is based on the Hurchalla's method for computing the
  // multiplicative inverse modulo a power of two. For better understanding
  // the implementation, the following paper is recommended:
  // J. Hurchalla, "An Improved Integer Multiplicative Inverse (modulo 2ʷ)",
  // https://arxiv.org/pdf/2204.04342.pdf
  constexpr static uint64_t Invert62(const uint64_t value) {
    uint64_t x = 3 * value ^ 2;
    uint64_t y = 1 - x * value;
    x *= y + 1;
    y *= y;
    x *= y + 1;
    y *= y;
    x *= y + 1;
    y *= y;
    return (x * (y + 1)) & LCInt::MASK;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_BYINVERTER_H_

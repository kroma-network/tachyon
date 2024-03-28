/**
 * @file
 * @brief Finite field operations for the %{type} curve.
 *
 * This header file defines operations and structures for manipulating elements of the finite field %{suffix} associated with the %{type} elliptic curve. 
 * It provides fundamental arithmetic operations necessary for elliptic curve cryptography on this curve such as addition, subtraction, 
 * and multiplication of field elements.
 * @example prime_field.cc
 */

// clang-format off
#include <stdint.h>

#include "tachyon/c/export.h"

/**
 * @struct tachyon_%{type}_%{suffix}
 * @brief Represents an element in the finite field %{suffix} for the %{type} curve.
 *
 * This structure is used to represent an element in the finite field %{suffix}, 
 * of the %{type} curve. It stores the element as an array of 64-bit limbs.
 */
struct tachyon_%{type}_%{suffix} {
  uint64_t limbs[%{limb_nums}];
};

%{extern_c_front}

/**
 * @brief Returns the zero element of the finite field %{suffix}.
 * @return The zero element in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_zero();

/**
 * @brief Returns the one element of the finite field %{suffix}.
 * @return The one element in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_one();

/**
 * @brief Generates a random element in the finite field %{suffix}.
 * @return A random element in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_random();

/**
 * @brief Adds two elements in the finite field %{suffix}.
 * @param a Pointer to the first operand.
 * @param b Pointer to the second operand.
 * @return The result of the addition a + b in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_add(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Subtracts one element from another in the finite field %{suffix}.
 * @param a Pointer to the minuend.
 * @param b Pointer to the subtrahend.
 * @return The result of the subtraction a - b in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sub(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Multiplies two elements in the finite field %{suffix}.
 * @param a Pointer to the first factor.
 * @param b Pointer to the second factor.
 * @return The result of the multiplication a * b in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_mul(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Divides one element by another in the finite field %{suffix}.
 * @param a Pointer to the dividend.
 * @param b Pointer to the divisor.
 * @return The result of the division a / b in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_div(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Negates an element in the finite field %{suffix}.
 * @param a Pointer to the element to negate.
 * @return The negation of a in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_neg(const tachyon_%{type}_%{suffix}* a);

/**
 * @brief Doubles an element in the finite field %{suffix}.
 * @param a Pointer to the element to double.
 * @return The result of doubling a in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_dbl(const tachyon_%{type}_%{suffix}* a);

/**
 * @brief Squares an element in the finite field %{suffix}.
 * @param a Pointer to the element to square.
 * @return The square of a in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_sqr(const tachyon_%{type}_%{suffix}* a);

/**
 * @brief Calculates the multiplicative inverse of an element in the finite field %{suffix}.
 * @param a Pointer to the element to invert.
 * @return The multiplicative inverse of a in %{suffix}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{suffix} tachyon_%{type}_%{suffix}_inv(const tachyon_%{type}_%{suffix}* a);

/**
 * @brief Checks if two elements in the finite field %{suffix} are equal.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a and b are equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_eq(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Checks if two elements in the finite field %{suffix} are not equal.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a and b are not equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_ne(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Checks if one element in the finite field %{suffix} is greater than another.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a is greater than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_gt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Checks if one element in the finite field %{suffix} is greater than or equal to another.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a is greater than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_ge(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Checks if one element in the finite field %{suffix} is less than another.
 * This function compares two elements `a` and `b` in the finite field %{suffix} to determine if `a` is strictly less than `b`.
 * @param a Pointer to the first element for comparison.
 * @param b Pointer to the second element for comparison.
 * @return True if `a` is strictly less than `b`, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_lt(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

/**
 * @brief Checks if one element in the finite field %{suffix} is less than or equal to another.
 * This function compares two elements `a` and `b` in the finite field %{suffix} to determine if `a` is less than or equal to `b`. It provides an essential comparison operation for cryptographic algorithms involving finite fields.
 * @param a Pointer to the first element for comparison.
 * @param b Pointer to the second element for comparison.
 * @return True if `a` is less than or equal to `b`, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_%{suffix}_le(const tachyon_%{type}_%{suffix}* a, const tachyon_%{type}_%{suffix}* b);

// clang-format on

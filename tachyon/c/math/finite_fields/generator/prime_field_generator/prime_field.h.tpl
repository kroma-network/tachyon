// clang-format off
%{if IsECPrimeField}
/**
 * @file %{class_name}.h
 * @brief Finite field operations for the %{curve} curve.
 *
 * This header file defines operations and structures for manipulating elements of the finite field %{display_name} associated with the %{curve} elliptic curve.
 * It provides fundamental arithmetic operations necessary for elliptic curve cryptography on this curve such as addition, subtraction,
 * and multiplication of field elements.
 * @example prime_field.cc
 */
%{endif IsECPrimeField}
%{if IsSmallPrimeField}
/**
 * @file %{class_name}.h
 * @brief Finite field operations for the %{display_name}
 *
 * This header file defines operations and structures for manipulating elements of the finite field %{display_name}.
 * It provides fundamental arithmetic operations necessary such as addition, subtraction and multiplication of field elements.
 * @example prime_field.cc
 */
%{endif IsSmallPrimeField}

#include <stdint.h>

#include "tachyon/c/export.h"

%{if IsECPrimeField}
/**
 * @struct tachyon_%{class_name}
 * @brief Represents an element in the finite field %{display_name} for the %{curve} curve.
 *
 * This structure is used to represent an element in the finite field %{display_name},
 * of the %{curve} curve. It stores the element as an array of 64-bit limbs.
 */
struct tachyon_%{class_name} {
  uint64_t limbs[%{limb_nums}];
};
%{endif IsECPrimeField}
%{if IsSmallPrimeField}
/**
 * @struct tachyon_%{class_name}
 * @brief Represents an element in the finite field %{display_name}.
 *
 * This structure is used to represent an element in the finite field %{display_name},
 * It stores the element as a 32-bit value.
 */
struct tachyon_%{class_name} {
  uint32_t value;
};
%{endif IsSmallPrimeField}

%{extern_c_front}

/**
 * @brief Returns the zero element of the finite field %{display_name}.
 * @return The zero element in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_zero();

/**
 * @brief Returns the one element of the finite field %{display_name}.
 * @return The one element in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_one();

/**
 * @brief Returns the minus one element of the finite field %{display_name}.
 * @return The one element in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_minus_one();

/**
 * @brief Generates a random element in the finite field %{display_name}.
 * @return A random element in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_random();

/**
 * @brief Adds two elements in the finite field %{display_name}.
 * @param a Pointer to the first operand.
 * @param b Pointer to the second operand.
 * @return The result of the addition a + b in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_add(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Subtracts one element from another in the finite field %{display_name}.
 * @param a Pointer to the minuend.
 * @param b Pointer to the subtrahend.
 * @return The result of the subtraction a - b in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_sub(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Multiplies two elements in the finite field %{display_name}.
 * @param a Pointer to the first factor.
 * @param b Pointer to the second factor.
 * @return The result of the multiplication a * b in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_mul(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Divides one element by another in the finite field %{display_name}.
 * @param a Pointer to the dividend.
 * @param b Pointer to the divisor.
 * @return The result of the division a / b in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_div(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Negates an element in the finite field %{display_name}.
 * @param a Pointer to the element to negate.
 * @return The negation of a in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_neg(const tachyon_%{class_name}* a);

/**
 * @brief Doubles an element in the finite field %{display_name}.
 * @param a Pointer to the element to double.
 * @return The result of doubling a in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_dbl(const tachyon_%{class_name}* a);

/**
 * @brief Squares an element in the finite field %{display_name}.
 * @param a Pointer to the element to square.
 * @return The square of a in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_sqr(const tachyon_%{class_name}* a);

/**
 * @brief Calculates the multiplicative inverse of an element in the finite field %{display_name}.
 * @param a Pointer to the element to invert.
 * @return The multiplicative inverse of a in %{display_name}.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_inv(const tachyon_%{class_name}* a);

/**
 * @brief Checks if two elements in the finite field %{display_name} are equal.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a and b are equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_eq(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if two elements in the finite field %{display_name} are not equal.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a and b are not equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_ne(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one element in the finite field %{display_name} is greater than another.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a is greater than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_gt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one element in the finite field %{display_name} is greater than or equal to another.
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @return True if a is greater than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_ge(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one element in the finite field %{display_name} is less than another.
 * This function compares two elements `a` and `b` in the finite field %{display_name} to determine if `a` is strictly less than `b`.
 * @param a Pointer to the first element for comparison.
 * @param b Pointer to the second element for comparison.
 * @return True if `a` is strictly less than `b`, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_lt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one element in the finite field %{display_name} is less than or equal to another.
 * This function compares two elements `a` and `b` in the finite field %{display_name} to determine if `a` is less than or equal to `b`. It provides an essential comparison operation for cryptographic algorithms involving finite fields.
 * @param a Pointer to the first element for comparison.
 * @param b Pointer to the second element for comparison.
 * @return True if `a` is less than or equal to `b`, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_le(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

// clang-format on

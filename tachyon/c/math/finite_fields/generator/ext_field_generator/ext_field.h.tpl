// clang-format off
/**
 * @file
 * @brief Defines operations over the finite field extension %{display_name}.
 *
 * This header file specifies the structure and operations for elements of the extended finite field %{display_name},
 * which is built over the base field %{base_field_display_name} as part of the %{curve} elliptic curve cryptographic operations.
 * It includes basic arithmetic operations such as addition, subtraction, multiplication, and inversion,
 * as well as comparison operations.
 * @example extension_field.cc
 */

#include <stdint.h>

#include "tachyon/c/export.h"
#include "%{c_base_field_hdr}"

/**
 * @struct tachyon_%{class_name}
 * @brief Represents an element in the finite field extension %{display_name}.
 *
 * This structure models an element in %{display_name}, which is constructed over the base field %{base_field_display_name}.
 * It is used for cryptographic operations that require arithmetic in an extended finite field.
 */
struct tachyon_%{class_name} {
  tachyon_%{base_field_class_name} c0;
  tachyon_%{base_field_class_name} c1;
%{if IsCubicExtension}
  tachyon_%{base_field_class_name} c2;
%{endif IsCubicExtension}
%{if IsQuarticExtension}
  tachyon_%{base_field_class_name} c2;
  tachyon_%{base_field_class_name} c3;
%{endif IsQuarticExtension}
};

%{extern_c_front}

/**
 * @brief Returns an %{display_name} element with the value zero.
 *
 * @return An %{display_name} element representing zero.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_zero();

/**
 * @brief Returns an %{display_name} element with the value one.
 *
 * @return An %{display_name} element representing one.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_one();

/**
 * @brief Returns an %{display_name} element with the value minus one.
 *
 * @return An %{display_name} element representing minus one.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_minus_one();

/**
 * @brief Generates a random %{display_name} element.
 *
 * @return A randomly generated %{display_name} element.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_random();

/**
 * @brief Doubles an %{display_name} element.
 *
 * @param a Pointer to the %{display_name} element to double.
 * @return The doubled %{display_name} element.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_dbl(const tachyon_%{class_name}* a);

/**
 * @brief Negates an %{display_name} element.
 *
 * @param a Pointer to the %{display_name} element to negate.
 * @return The negated %{display_name} element.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_neg(const tachyon_%{class_name}* a);

/**
 * @brief Squares an %{display_name} element.
 *
 * @param a Pointer to the %{display_name} element to square.
 * @return The squared %{display_name} element.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_sqr(const tachyon_%{class_name}* a);

/**
 * @brief Inverts an %{display_name} element.
 *
 * @param a Pointer to the %{display_name} element to invert.
 * @return The inverted %{display_name} element. If the input is zero, the program will be killed.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_inv(const tachyon_%{class_name}* a);

/**
 * @brief Adds two %{display_name} elements.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return The sum a + b.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_add(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Subtracts one %{display_name} element from another.
 *
 * @param a Pointer to the %{display_name} element from which to subtract.
 * @param b Pointer to the %{display_name} element to subtract.
 * @return The difference a - b.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_sub(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Multiplies two %{display_name} elements.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return The product a * b.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_mul(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Divides one %{display_name} element by another.
 *
 * @param a Pointer to the %{display_name} element to divide.
 * @param b Pointer to the %{display_name} element to divide by.
 * @return The quotient a / b. If b is zero, the program will be killed.
 */
TACHYON_C_EXPORT tachyon_%{class_name} tachyon_%{class_name}_div(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if two %{display_name} elements are equal.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a and b are equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_eq(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if two %{display_name} elements are not equal.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a and b are not equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_ne(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one %{display_name} element is greater than another.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a is greater than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_gt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one %{display_name} element is greater than or equal to another.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a is greater than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_ge(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one %{display_name} element is less than another.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a is less than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_lt(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);

/**
 * @brief Checks if one %{display_name} element is less than or equal to another.
 *
 * @param a Pointer to the first %{display_name} element.
 * @param b Pointer to the second %{display_name} element.
 * @return True if a is less than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{class_name}_le(const tachyon_%{class_name}* a, const tachyon_%{class_name}* b);
// clang-format on

/**
 * @file
 * @brief Defines operations over the finite field extension Fq%{degree} for the %{type} curve.
 *
 * This header file specifies the structure and operations for elements of the extended finite field Fq%{degree},
 * which is built over the base field Fq%{base_field_degree} as part of the %{type} elliptic curve cryptographic operations.
 * It includes basic arithmetic operations such as addition, subtraction, multiplication, and inversion,
 * as well as comparison operations.
 * @example extension_field.cc
 */

// clang-format off
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fq%{base_field_degree}.h"

/**
 * @struct tachyon_%{type}_fq%{degree}
 * @brief Represents an element in the finite field extension Fq%{degree} for the %{type} curve.
 *
 * This structure models an element in Fq%{degree}, which is constructed over the base field Fq%{base_field_degree}.
 * It is used for cryptographic operations that require arithmetic in an extended finite field.
 */
struct tachyon_%{type}_fq%{degree} {
  tachyon_%{type}_fq%{base_field_degree} c0;
  tachyon_%{type}_fq%{base_field_degree} c1;
%{if IsCubicExtension}
  tachyon_%{type}_fq%{base_field_degree} c2;
%{endif IsCubicExtension}
};

%{extern_c_front}

/**
 * @brief Initializes an Fq%{degree} element to zero.
 *
 * @return An Fq%{degree} element representing zero.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_zero();

/**
 * @brief Initializes an Fq%{degree} element to one.
 *
 * @return An Fq%{degree} element representing one.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_one();

/**
 * @brief Generates a random Fq%{degree} element.
 *
 * @return A randomly generated Fq%{degree} element.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_random();

/**
 * @brief Doubles an Fq%{degree} element.
 *
 * @param a Pointer to the Fq%{degree} element to double.
 * @return The doubled Fq%{degree} element.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_dbl(const tachyon_%{type}_fq%{degree}* a);

/**
 * @brief Negates an Fq%{degree} element.
 *
 * @param a Pointer to the Fq%{degree} element to negate.
 * @return The negated Fq%{degree} element.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_neg(const tachyon_%{type}_fq%{degree}* a);

/**
 * @brief Squares an Fq%{degree} element.
 *
 * @param a Pointer to the Fq%{degree} element to square.
 * @return The squared Fq%{degree} element.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sqr(const tachyon_%{type}_fq%{degree}* a);

/**
 * @brief Inverts an Fq%{degree} element.
 *
 * @param a Pointer to the Fq%{degree} element to invert.
 * @return The inverted Fq%{degree} element. If the input is zero, the program will be killed.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_inv(const tachyon_%{type}_fq%{degree}* a);

/**
 * @brief Adds two Fq%{degree} elements.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return The sum a + b.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_add(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Subtracts one Fq%{degree} element from another.
 *
 * @param a Pointer to the Fq%{degree} element from which to subtract.
 * @param b Pointer to the Fq%{degree} element to subtract.
 * @return The difference a - b.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_sub(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Multiplies two Fq%{degree} elements.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return The product a * b.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_mul(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Divides one Fq%{degree} element by another.
 *
 * @param a Pointer to the Fq%{degree} element to divide.
 * @param b Pointer to the Fq%{degree} element to divide by.
 * @return The quotient a / b. If b is zero, the program will be killed.
 */
TACHYON_C_EXPORT tachyon_%{type}_fq%{degree} tachyon_%{type}_fq%{degree}_div(const tachyon_%{type}_fq%{degree}* a,
                                            const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if two Fq%{degree} elements are equal.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a and b are equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_eq(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if two Fq%{degree} elements are not equal.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a and b are not equal, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_ne(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if one Fq%{degree} element is greater than another.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a is greater than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_gt(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if one Fq%{degree} element is greater than or equal to another.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a is greater than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_ge(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if one Fq%{degree} element is less than another.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a is less than b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_lt(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);

/**
 * @brief Checks if one Fq%{degree} element is less than or equal to another.
 *
 * @param a Pointer to the first Fq%{degree} element.
 * @param b Pointer to the second Fq%{degree} element.
 * @return True if a is less than or equal to b, false otherwise.
 */
TACHYON_C_EXPORT bool tachyon_%{type}_fq%{degree}_le(const tachyon_%{type}_fq%{degree}* a,
                                           const tachyon_%{type}_fq%{degree}* b);
// clang-format on

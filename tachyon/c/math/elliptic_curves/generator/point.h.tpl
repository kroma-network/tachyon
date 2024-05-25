/**
 * @file
 * @brief Defines structures and operations for elliptic curve points in %{g1_or_g2} group using %{type} curve.
 *
 * This header file provides definitions for different representations of elliptic curve points (affine, projective, Jacobian, and XYZZ coordinates)
 * on the %{g1_or_g2} group of the %{type} curve, along with functions for point operations such as addition, subtraction, negation, and comparison.
 * The operations are optimized for efficiency and are crucial for cryptographic algorithms implemented using elliptic curves.
 */

// clang-format off
#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{base_field}.h"

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_affine
 * @brief Represents an affine point on the %{type} %{g1_or_g2} curve.
 *
 * This structure models an affine point on the %{type} curve with x and y coordinates.
 *
 * @example affine_point.cc
 */
struct tachyon_%{type}_%{g1_or_g2}_affine {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_projective
 * @brief Represents a point on the %{type} %{g1_or_g2} curve in projective coordinates.
 *
 * Projective coordinates are useful for elliptic curve operations as they
 * allow for operations without division, improving efficiency.
 *
 * @example projective_point.cc
 */
struct tachyon_%{type}_%{g1_or_g2}_projective {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_jacobian
 * @brief Represents a point on the %{type} %{g1_or_g2} curve in jacobian coordinates.
 *
 * Jacobian coordinates are a type of projective coordinates that are often
 * used for internal calculations to speed up the arithmetic on elliptic curves.
 * @example jacobian_point.cc
 */
struct tachyon_%{type}_%{g1_or_g2}_jacobian {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_xyzz
 * @brief Represents an extended point on the %{type} %{g1_or_g2} curve with precomputed z squared and cubed values.
 *
 * XYZZ coordinates are used for optimizing certain elliptic curve operations by precomputing
 * and storing z^2 and z^3 along with the standard coordinates.
 * @example point_xyzz.cc
 */
struct tachyon_%{type}_%{g1_or_g2}_xyzz {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} zz;
  tachyon_%{type}_%{base_field} zzz;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_point2
 * @brief Represents a simple 2D point on the %{type} %{g1_or_g2} curve in affine coordinates.
 *
 * This structure can be used for operations where only x and y coordinates are required,
 * and the point is known not to be at infinity.
 */
struct tachyon_%{type}_%{g1_or_g2}_point2 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_point3
 * @brief Represents a 3D point on the %{type} %{g1_or_g2} curve, potentially in projective or jacobian coordinates.
 *
 * This structure is a general representation that can be used in various contexts where
 * a third coordinate is necessary for the calculations.
 */
struct tachyon_%{type}_%{g1_or_g2}_point3 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
};

/**
 * @struct tachyon_%{type}_%{g1_or_g2}_point4
 * @brief Represents a point on the %{type} %{g1_or_g2} curve with an additional auxiliary coordinate.
 *
 * This structure is designed for specific algorithms that require an extra coordinate for
 * efficient computation, such as certain multi-scalar multiplication techniques.
 */
struct tachyon_%{type}_%{g1_or_g2}_point4 {
  tachyon_%{type}_%{base_field} x;
  tachyon_%{type}_%{base_field} y;
  tachyon_%{type}_%{base_field} z;
  tachyon_%{type}_%{base_field} w;
};

%{extern_c_front}

/**
 * @brief Initializes the %{type} %{g1_or_g2} curve.
 *
 * This function performs any necessary initializations for the %{type} %{g1_or_g2} curve operations.
 */
TACHYON_C_EXPORT void tachyon_%{type}_%{g1_or_g2}_init();

/**
 * @brief Returns the zero point in affine coordinates on the %{type} %{g1_or_g2} curve.
 * @return The zero point in affine coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_zero();

/**
 * @brief Returns the zero point in projective coordinates on the %{type} %{g1_or_g2} curve.
 * @return The zero point in projective coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_zero();

/**
 * @brief Returns the zero point in jacobian coordinates on the %{type} %{g1_or_g2} curve.
 * @return The zero point in jacobian coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_zero();

/**
 * @brief Returns the zero point in xyzz coordinates on the %{type} %{g1_or_g2} curve.
 * @return The zero point in xyzz coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_zero();

/**
 * @brief Returns the generator of the affine group %{g1_or_g2}.
 *
 * This function returns the predefined generator point of the affine group
 * %{g1_or_g2}, which is a fundamental element for elliptic curve operations.
 *
 * @return The generator point of the affine group %{g1_or_g2}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_generator();

/**
 * @brief Returns the generator of the projective group %{g1_or_g2}.
 *
 * This function returns the predefined generator point of the projective group
 * %{g1_or_g2}, which is used in elliptic curve cryptographic algorithms where
 * projective coordinates provide efficiency improvements.
 *
 * @return The generator point of the projective group %{g1_or_g2}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_generator();

/**
 * @brief Returns the generator of the jacobian group %{g1_or_g2}.
 *
 * This function returns the predefined generator point of the jacobian group
 * %{g1_or_g2}. jacobian coordinates are used in elliptic curve cryptography to
 * optimize point addition and doubling operations.
 *
 * @return The generator point of the jacobian group %{g1_or_g2}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_generator();

/**
 * @brief Returns the generator of the XYZZ group %{g1_or_g2}.
 *
 * This function returns the predefined generator point of the XYZZ group
 * %{g1_or_g2}, where the ZZ and ZZZ values are precomputed to optimize certain
 * elliptic curve operations. It's an advanced representation for specific algorithms.
 *
 * @return The generator point of the XYZZ group %{g1_or_g2}.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_generator();

/**
 * @brief Generates a random affine point on the %{type} %{g1_or_g2} curve.
 * @return A random affine point on the %{g1_or_g2} curve.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_random();

/**
 * @brief Generates a random projective point on the %{type} %{g1_or_g2} curve.
 * @return A random projective point on the %{g1_or_g2} curve.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_random();

/**
 * @brief Generates a random jacobian point on the %{type} %{g1_or_g2} curve.
 * @return A random jacobian point on the %{g1_or_g2} curve.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_random();

/**
 * @brief Generates a random xyzz point on the %{type} %{g1_or_g2} curve.
 * @return A random xyzz point on the %{g1_or_g2} curve.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_random();

/**
 * @brief Adds two affine points on the %{type} %{g1_or_g2} curve. The return type for the addition of affine points is jacobian.
 * @param a Pointer to the first affine point.
 * @param b Pointer to the second affine point.
 * @return The result of adding points a and b, represented in jacobian coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_add(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Adds two projective points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the first projective point.
 * @param b Pointer to the second projective point.
 * @return The result of the addition, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

/**
 * @brief Computes the addition of a projective point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the projective point.
 * @param b Pointer to the affine point.
 * @return The result of the addition, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_add_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Adds two jacobian points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the first jacobian point.
 * @param b Pointer to the second jacobian point.
 * @return The result of the addition, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

/**
 * @brief Computes the addition of a jacobian point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the jacobian point.
 * @param b Pointer to the affine point.
 * @return The result of the addition, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_add_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Adds two xyzz points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the first xyzz point.
 * @param b Pointer to the second xyzz point.
 * @return The result of the addition, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

/**
 * @brief Computes the addition of a xyzz point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the xyzz point.
 * @param b Pointer to the affine point.
 * @return The result of the addition, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_add_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Computes the subtraction of two affine points on the %{type} %{g1_or_g2} curve. The return type for the addition of affine points is jacobian.
 * @param a Pointer to the affine point from which to subtract.
 * @param b Pointer to the affine point to subtract.
 * @return The result of the subtraction, as a affine point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_sub(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Computes the subtraction of two projective points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the projective point from which to subtract.
 * @param b Pointer to the projective point to subtract.
 * @return The result of the subtraction, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

/**
 * @brief Computes the subtraction of a projective point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the projective point from which to subtract.
 * @param b Pointer to the affine point to subtract.
 * @return The result of the subtraction, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Computes the subtraction of two jacobian points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the jacobian point from which to subtract.
 * @param b Pointer to the jacobian point to subtract.
 * @return The result of the subtraction, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

/**
 * @brief Computes the subtraction of a jacobian point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the jacobian point from which to subtract.
 * @param b Pointer to the affine point to subtract.
 * @return The result of the subtraction, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Computes the subtraction of two xyzz points on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the xyzz point from which to subtract.
 * @param b Pointer to the xyzz point to subtract.
 * @return The result of the subtraction, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

/**
 * @brief Computes the subtraction of a xyzz point and an affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the xyzz point from which to subtract.
 * @param b Pointer to the affine point to subtract.
 * @return The result of the subtraction, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_sub_mixed(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Computes the negation of a affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the affine point to negate.
 * @return The negation of the point, as a affine point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine tachyon_%{type}_%{g1_or_g2}_affine_neg(const tachyon_%{type}_%{g1_or_g2}_affine* a);

/**
 * @brief Computes the negation of a projective point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the projective point to negate.
 * @return The negation of the point, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_neg(const tachyon_%{type}_%{g1_or_g2}_projective* a);

/**
 * @brief Computes the negation of a jacobian point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the jacobian point to negate.
 * @return The negation of the point, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_neg(const tachyon_%{type}_%{g1_or_g2}_jacobian* a);

/**
 * @brief Computes the negation of a xyzz point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the xyzz point to negate.
 * @return The negation of the point, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_neg(const tachyon_%{type}_%{g1_or_g2}_xyzz* a);

/**
 * @brief Doubles a affine point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the affine point to double.
 * @return The result of doubling the point, as a affine point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_affine_dbl(const tachyon_%{type}_%{g1_or_g2}_affine* a);

/**
 * @brief Doubles a projective point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the projective point to double.
 * @return The result of doubling the point, as a projective point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective tachyon_%{type}_%{g1_or_g2}_projective_dbl(const tachyon_%{type}_%{g1_or_g2}_projective* a);

/**
 * @brief Doubles a jacobian point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the jacobian point to double.
 * @return The result of doubling the point, as a jacobian point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian tachyon_%{type}_%{g1_or_g2}_jacobian_dbl(const tachyon_%{type}_%{g1_or_g2}_jacobian* a);

/**
 * @brief Doubles a xyzz point on the %{type} %{g1_or_g2} curve.
 * @param a Pointer to the xyzz point to double.
 * @return The result of doubling the point, as a xyzz point.
 */
TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz tachyon_%{type}_%{g1_or_g2}_xyzz_dbl(const tachyon_%{type}_%{g1_or_g2}_xyzz* a);

/**
 * @brief Checks if two affine points on the %{type} %{g1_or_g2} curve are equal.
 * @param a Pointer to the first affine point.
 * @param b Pointer to the second affine point.
 * @return True if the points are equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine_eq(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Checks if two projective points on the %{type} %{g1_or_g2} curve are equal.
 * @param a Pointer to the first projective point.
 * @param b Pointer to the second projective point.
 * @return True if the points are equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective_eq(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

/**
 * @brief Checks if two jacobian points on the %{type} %{g1_or_g2} curve are equal.
 * @param a Pointer to the first jacobian point.
 * @param b Pointer to the second jacobian point.
 * @return True if the points are equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian_eq(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

/**
 * @brief Checks if two xyzz points on the %{type} %{g1_or_g2} curve are equal.
 * @param a Pointer to the first xyzz point.
 * @param b Pointer to the second xyzz point.
 * @return True if the points are equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz_eq(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);

/**
 * @brief Checks if two affine points on the %{type} %{g1_or_g2} curve are not equal.
 * @param a Pointer to the first affine point.
 * @param b Pointer to the second affine point.
 * @return True if the points are not equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_affine_ne(const tachyon_%{type}_%{g1_or_g2}_affine* a, const tachyon_%{type}_%{g1_or_g2}_affine* b);

/**
 * @brief Checks if two projective points on the %{type} %{g1_or_g2} curve are not equal.
 * @param a Pointer to the first projective point.
 * @param b Pointer to the second projective point.
 * @return True if the points are not equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_projective_ne(const tachyon_%{type}_%{g1_or_g2}_projective* a, const tachyon_%{type}_%{g1_or_g2}_projective* b);

/**
 * @brief Checks if two jacobian points on the %{type} %{g1_or_g2} curve are not equal.
 * @param a Pointer to the first jacobian point.
 * @param b Pointer to the second jacobian point.
 * @return True if the points are not equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_jacobian_ne(const tachyon_%{type}_%{g1_or_g2}_jacobian* a, const tachyon_%{type}_%{g1_or_g2}_jacobian* b);

/**
 * @brief Checks if two xyzz points on the %{type} %{g1_or_g2} curve are not equal.
 * @param a Pointer to the first xyzz point.
 * @param b Pointer to the second xyzz point.
 * @return True if the points are not equal, false otherwise.
 */
bool TACHYON_C_EXPORT tachyon_%{type}_%{g1_or_g2}_xyzz_ne(const tachyon_%{type}_%{g1_or_g2}_xyzz* a, const tachyon_%{type}_%{g1_or_g2}_xyzz* b);
// clang-format on

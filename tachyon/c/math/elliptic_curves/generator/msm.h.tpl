// clang-format off
/**
 * @file
 * @brief Multi-scalar multiplication (MSM) operations for %{type} curve points in a G1 group.
 *
 * This header file defines the interface for performing multi-scalar multiplication
 * operations on points of the G1 group of the %{type} curve. MSM is a crucial operation
 * in cryptographic schemes, enabling efficient computation of the sum of scalar
 * multiplications of points.
 *
 * @example msm.cc
 */

#include <stddef.h>
#include <stdint.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/fr.h"
#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/g1.h"

typedef struct tachyon_%{type}_g1_msm* tachyon_%{type}_g1_msm_ptr;

%{extern_c_front}

/**
 * @brief Creates a new MSM context for the G1 group with a specified polynomial degree.
 * @param degree The degree of the polynomial for the MSM operation. Currently not used.
 * @return A pointer to the newly created MSM context.
 */
TACHYON_C_EXPORT tachyon_%{type}_g1_msm_ptr tachyon_%{type}_g1_create_msm(uint8_t degree);

/**
 * @brief Destroys an MSM context, freeing its resources.
 * @param ptr The pointer to the MSM context to destroy.
 */
TACHYON_C_EXPORT void tachyon_%{type}_g1_destroy_msm(tachyon_%{type}_g1_msm_ptr ptr);

/**
 * @brief Computes MSM using projective bases and scalars.
 * @param ptr The MSM context.
 * @param bases Array of projective points to be multiplied by corresponding scalars.
 * @param scalars Array of scalars for the multiplication.
 * @param size The number of points and scalars (must be the same).
 * @return A pointer to the result of the MSM operation in Jacobian coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_point2_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_point2* bases,
    const tachyon_%{type}_fr* scalars, size_t size);

/**
 * @brief Computes MSM using affine bases and scalars.
 * @param ptr The MSM context.
 * @param bases Array of affine points to be multiplied by corresponding scalars.
 * @param scalars Array of scalars for the multiplication.
 * @param size The number of points and scalars (must be the same).
 * @return A pointer to the result of the MSM operation in Jacobian coordinates.
 */
TACHYON_C_EXPORT tachyon_%{type}_g1_jacobian* tachyon_%{type}_g1_affine_msm(
    tachyon_%{type}_g1_msm_ptr ptr, const tachyon_%{type}_g1_affine* bases,
    const tachyon_%{type}_fr* scalars, size_t size);
// clang-format off

/**
 * @file baby_bear_row_major_matrix.h
 * @brief Row major matrix operations for baby bear.
 *
 * This header file defines the structure and API for row major matrix for
 * baby bear. This includes creation, cloning, and destruction of row major
 * matrix structures, which are fundamental to various cryptographic protocols
 * and algorithms implemented on this field.
 */

#ifndef TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_H_
#define TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_H_

#include <stddef.h>

#include "tachyon/c/export.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"

/**
 * @struct tachyon_baby_bear_row_major_matrix
 * @brief Represents a row major matrix for baby bear.
 *
 * This structure encapsulates a row major matrix, offering efficient
 * storage and manipulation for cryptographic computations.
 */
struct tachyon_baby_bear_row_major_matrix {};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a new instance of a row major matrix.
 *
 * Allocates and initializes a new row major matrix structure for use
 * in cryptographic algorithms.
 *
 * @param ptr A pointer to the matrix data.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @return A pointer to the newly created matrix structure.
 */
TACHYON_C_EXPORT tachyon_baby_bear_row_major_matrix*
tachyon_baby_bear_row_major_matrix_create(tachyon_baby_bear* ptr, size_t rows,
                                          size_t cols);

/**
 * @brief Clones a row major matrix.
 *
 * Creates a deep copy of the given row major matrix.
 *
 * @param matrix A const pointer to the matrix to be cloned.
 * @return A pointer to the cloned matrix structure.
 */
TACHYON_C_EXPORT tachyon_baby_bear_row_major_matrix*
tachyon_baby_bear_row_major_matrix_clone(
    const tachyon_baby_bear_row_major_matrix* matrix);

/**
 * @brief Destroys a row major matrix.
 *
 * Frees the memory allocated for a row major matrix structure,
 * effectively destroying it.
 *
 * @param matrix A pointer to the matrix to be destroyed.
 */
TACHYON_C_EXPORT void tachyon_baby_bear_row_major_matrix_destroy(
    tachyon_baby_bear_row_major_matrix* matrix);

/**
 * @brief Returns the number of rows in the row major matrix.
 *
 * @param matrix A const pointer to the matrix.
 * @return The number of rows in the row major matrix.
 */
TACHYON_C_EXPORT size_t tachyon_baby_bear_row_major_matrix_get_rows(
    const tachyon_baby_bear_row_major_matrix* matrix);

/**
 * @brief Returns the number of columns in the row major matrix.
 *
 * @param matrix A const pointer to the matrix.
 * @return The number of columns in the row major matrix.
 */
TACHYON_C_EXPORT size_t tachyon_baby_bear_row_major_matrix_get_cols(
    const tachyon_baby_bear_row_major_matrix* matrix);

/**
 * @brief Returns the matrix element at a given row and column index.
 *
 * @param matrix A const pointer to the matrix.
 * @param row The row index.
 * @param col The column index.
 * @return The element of the row major matrix at a given row and column index.
 */
TACHYON_C_EXPORT tachyon_baby_bear
tachyon_baby_bear_row_major_matrix_get_element(
    const tachyon_baby_bear_row_major_matrix* matrix, size_t row, size_t col);

/**
 * @brief Returns a const pointer to the matrix data.
 *
 * @param matrix A const pointer to the matrix.
 * @return A const pointer to the matrix data.
 */
TACHYON_C_EXPORT const tachyon_baby_bear*
tachyon_baby_bear_row_major_matrix_get_const_data_ptr(
    const tachyon_baby_bear_row_major_matrix* matrix);

/**
 * @brief Returns a pointer to the matrix data.
 *
 * @param matrix A pointer to the matrix.
 * @return A pointer to the matrix data.
 */
TACHYON_C_EXPORT tachyon_baby_bear*
tachyon_baby_bear_row_major_matrix_get_data_ptr(
    tachyon_baby_bear_row_major_matrix* matrix);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TACHYON_C_MATH_MATRIX_BABY_BEAR_ROW_MAJOR_MATRIX_H_

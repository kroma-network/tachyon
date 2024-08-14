#include "tachyon/c/math/matrix/baby_bear_row_major_matrix.h"

#include "gtest/gtest.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

using F = BabyBear;

class RowMajorMatrixTest : public FiniteFieldTest<F> {
 public:
  void SetUp() override {
    cpp_matrix_ = RowMajorMatrix<F>::Random(5, 5);
    matrix_ = c::base::c_cast(&cpp_matrix_);
  }

 protected:
  RowMajorMatrix<F> cpp_matrix_;
  tachyon_baby_bear_row_major_matrix* matrix_;
};

}  // namespace

TEST_F(RowMajorMatrixTest, APIs) {
  EXPECT_EQ(cpp_matrix_,
            c::base::native_cast(*tachyon_baby_bear_row_major_matrix_create(
                c::base::c_cast(cpp_matrix_.data()), 5, 5)));

  tachyon_baby_bear_row_major_matrix* cloned =
      tachyon_baby_bear_row_major_matrix_clone(matrix_);
  tachyon_baby_bear_row_major_matrix_destroy(cloned);

  EXPECT_EQ(5, tachyon_baby_bear_row_major_matrix_get_rows(matrix_));
  EXPECT_EQ(5, tachyon_baby_bear_row_major_matrix_get_cols(matrix_));

  EXPECT_EQ(cpp_matrix_(2, 3),
            c::base::native_cast(
                tachyon_baby_bear_row_major_matrix_get_element(matrix_, 2, 3)));

  EXPECT_EQ(
      cpp_matrix_.data(),
      c::base::native_cast(
          tachyon_baby_bear_row_major_matrix_get_const_data_ptr(matrix_)));
  EXPECT_EQ(cpp_matrix_.data(),
            c::base::native_cast(
                tachyon_baby_bear_row_major_matrix_get_data_ptr(matrix_)));
}

}  // namespace tachyon::math

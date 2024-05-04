#include "tachyon/math/matrix/matrix_utils.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/prime_field_num_traits.h"

namespace tachyon::math {

class MatrixTypesTest : public FiniteFieldTest<GF7> {};

TEST_F(MatrixTypesTest, Circulant) {
  math::Matrix<GF7> circulant =
      math::MakeCirculant(math::Vector<GF7>{{GF7(2), GF7(3), GF7(4)}});
  math::Matrix<GF7> expected{{
      {GF7(2), GF7(4), GF7(3)},
      {GF7(3), GF7(2), GF7(4)},
      {GF7(4), GF7(3), GF7(2)},
  }};
  EXPECT_EQ(circulant, expected);
}

}  // namespace tachyon::math

#include "tachyon/crypto/commitments/fri/fri_config.h"

#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {
namespace {

using F = math::BabyBear;
using ExtF = math::BabyBear4;

class FriConfigTest : public math::FiniteFieldTest<ExtF> {};

}  // namespace

TEST_F(FriConfigTest, FoldMatrix) {
  math::RowMajorMatrix<ExtF> inside(4, 2);
  inside << ExtF(F(7)), ExtF(F(1)), ExtF(F(2)), ExtF(F(3)), ExtF(F(5)),
      ExtF(F(3)), ExtF(F(2)), ExtF(F(5));
  std::vector<ExtF> result = FoldMatrix(ExtF(F(28)), inside);
  EXPECT_EQ(result[0], ExtF(F(88)));
  EXPECT_EQ(result[1], ExtF(F(1045105093)));
  EXPECT_EQ(result[2], ExtF(F(111548335)));
  EXPECT_EQ(result[3], ExtF(F(1448238559)));
}

}  // namespace tachyon::crypto

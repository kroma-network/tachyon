#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(FrTest, Init) {
  bn254::Fr::Init();
  SUCCEED();
}

}  // namespace math
}  // namespace tachyon

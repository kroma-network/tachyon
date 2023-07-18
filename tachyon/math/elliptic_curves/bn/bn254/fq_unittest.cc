#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(FqTest, Init) {
  bn254::Fq::Init();
  SUCCEED();
}

}  // namespace math
}  // namespace tachyon

#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(FqTest, Init) {
  bls12_381::Fq::Init();
  SUCCEED();
}

}  // namespace math
}  // namespace tachyon

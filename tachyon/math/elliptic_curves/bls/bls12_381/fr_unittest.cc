#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(FrTest, Init) {
  bls12_381::Fr::Init();
  SUCCEED();
}

}  // namespace math
}  // namespace tachyon

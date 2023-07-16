#include "tachyon/math/base/gmp/gmp_identities.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(GmpIdentities, MultiplicativeIdentity) {
  EXPECT_EQ(One<mpz_class>(), mpz_class(1));
  EXPECT_TRUE(IsOne<mpz_class>(mpz_class(1)));
  EXPECT_FALSE(IsOne<mpz_class>(mpz_class(0)));
}

TEST(GmpIdentities, AdditiveIdentity) {
  EXPECT_EQ(Zero<mpz_class>(), mpz_class(0));
  EXPECT_TRUE(IsZero<mpz_class>(mpz_class(0)));
  EXPECT_FALSE(IsZero<mpz_class>(mpz_class(1)));
}

}  // namespace math
}  // namespace tachyon
#ifndef TACHYON_MATH_BASE_GMP_GMP_IDENTITIES_H_
#define TACHYON_MATH_BASE_GMP_GMP_IDENTITIES_H_

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"

namespace tachyon {
namespace math {

template <>
class MultiplicativeIdentity<mpz_class> {
 public:
  static const mpz_class& One() {
    static base::NoDestructor<mpz_class> one(1);
    return *one;
  }

  static bool IsOne(const mpz_class& value) { return value == One(); }
};

template <>
class AdditiveIdentity<mpz_class> {
 public:
  static const mpz_class& Zero() {
    static base::NoDestructor<mpz_class> zero(0);
    return *zero;
  }

  static bool IsZero(const mpz_class& value) { return value == Zero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_GMP_GMP_IDENTITIES_H_
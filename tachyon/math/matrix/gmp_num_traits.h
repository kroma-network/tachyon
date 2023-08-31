#ifndef TACHYON_MATH_MATRIX_GMP_NUM_TRAITS_H_
#define TACHYON_MATH_MATRIX_GMP_NUM_TRAITS_H_

#include "third_party/eigen3/Eigen/Core"
#include "third_party/gmp/include/gmpxx.h"

namespace Eigen {

template <>
struct NumTraits<mpz_class> : GenericNumTraits<mpz_class> {
  enum {
    IsInteger = 1,
    IsSigned = 1,
    IsComplex = 0,
    RequireInitialization = 1,
    // NOTE(chokobole): I just used the same values defined at
    // https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html.
    ReadCost = 6,
    AddCost = 150,
    MulCost = 100,
  };
};

}  // namespace Eigen

#endif  // TACHYON_MATH_MATRIX_GMP_NUM_TRAITS_H_

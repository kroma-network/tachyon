#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_

#include "tachyon/math/base/field.h"

namespace tachyon {
namespace math {

template <typename F>
class PrimeFieldBase : public Field<F> {
 public:
  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  constexpr uint64_t operator%(uint64_t mod) const {
    const F* f = static_cast<const F*>(this);
    return f->Mod(mod);
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_BASE_H_

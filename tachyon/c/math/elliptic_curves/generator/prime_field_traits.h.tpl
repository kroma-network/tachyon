#include "tachyon/c/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"
#include "tachyon/cc/math/finite_fields/prime_field_traits_forward.h"
#include "tachyon/math/elliptic_curves/%{header_dir_name}/%{suffix}.h"

namespace tachyon::cc::math {

template <>
struct PrimeFieldTraits<tachyon_%{type}_%{suffix}> {
  using PrimeField = tachyon::math::%{type}::%{Suffix};
};

template <>
struct PrimeFieldTraits<tachyon::math::%{type}::%{Suffix}> {
  using CPrimeField = tachyon_%{type}_%{suffix};
};

}  // namespace tachyon::cc::math

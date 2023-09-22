#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FORWARD_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FORWARD_H_

namespace tachyon::math {

template <typename Config, typename SFINAE = void>
class PrimeField;

template <typename Config>
class PrimeFieldGmp;

template <typename Config>
class PrimeFieldGpu;

template <typename Config>
class PrimeFieldGpuDebug;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FORWARD_H_

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

template <typename T>
struct PrimeFieldTraits;

template <typename _Config>
struct PrimeFieldTraits<PrimeField<_Config>> {
  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGmp<_Config>> {
  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGpu<_Config>> {
  using Config = _Config;
};

template <typename _Config>
struct PrimeFieldTraits<PrimeFieldGpuDebug<_Config>> {
  using Config = _Config;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FORWARD_H_

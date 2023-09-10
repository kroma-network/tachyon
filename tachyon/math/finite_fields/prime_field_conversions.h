#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_

#include "tachyon/math/finite_fields/prime_field.h"
#if defined(TACHYON_GMP_BACKEND)
#include "tachyon/math/finite_fields/prime_field_gmp.h"
#endif  // defined(TACHYON_GMP_BACKEND)
#if TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_gpu.h"
#endif  // TACHYON_CUDA
#include "tachyon/math/finite_fields/prime_field_gpu_debug.h"

namespace tachyon::math {

template <typename SrcPrimeFieldTy, typename DstPrimeFieldTy>
struct PrimeFieldConversions;

template <typename DstPrimeFieldTy, typename SrcPrimeFieldTy>
constexpr DstPrimeFieldTy ConvertPrimeField(
    const SrcPrimeFieldTy& src_prime_field) {
  return PrimeFieldConversions<SrcPrimeFieldTy, DstPrimeFieldTy>::Convert(
      src_prime_field);
}

#if defined(TACHYON_GMP_BACKEND)
template <typename Config>
struct PrimeFieldConversions<PrimeField<Config>, PrimeFieldGmp<Config>> {
  constexpr static PrimeFieldGmp<Config> Convert(
      const PrimeField<Config>& src_prime_field) {
    return PrimeFieldGmp<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // defined(TACHYON_GMP_BACKEND)

#if TACHYON_CUDA
template <typename Config>
struct PrimeFieldConversions<PrimeField<Config>, PrimeFieldGpu<Config>> {
  constexpr static PrimeFieldGpu<Config> Convert(
      const PrimeField<Config>& src_prime_field) {
    return PrimeFieldGpu<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // TACHYON_CUDA

template <typename Config>
struct PrimeFieldConversions<PrimeField<Config>, PrimeFieldGpuDebug<Config>> {
  constexpr static PrimeFieldGpuDebug<Config> Convert(
      const PrimeField<Config>& src_prime_field) {
    return PrimeFieldGpuDebug<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};

#if defined(TACHYON_GMP_BACKEND)
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGmp<Config>, PrimeField<Config>> {
  constexpr static PrimeField<Config> Convert(
      const PrimeFieldGmp<Config>& src_prime_field) {
    return PrimeField<Config>::FromMontgomery(src_prime_field.ToMontgomery());
  }
};

#if TACHYON_CUDA
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGmp<Config>, PrimeFieldGpu<Config>> {
  constexpr static PrimeFieldGpu<Config> Convert(
      const PrimeFieldGmp<Config>& src_prime_field) {
    return PrimeFieldGpu<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // TACHYON_CUDA

template <typename Config>
struct PrimeFieldConversions<PrimeFieldGmp<Config>,
                             PrimeFieldGpuDebug<Config>> {
  constexpr static PrimeFieldGpuDebug<Config> Convert(
      const PrimeFieldGmp<Config>& src_prime_field) {
    return PrimeFieldGpuDebug<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // defined(TACHYON_GMP_BACKEND)

#if TACHYON_CUDA
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpu<Config>, PrimeField<Config>> {
  constexpr static PrimeField<Config> Convert(
      const PrimeFieldGpu<Config>& src_prime_field) {
    return PrimeField<Config>::FromMontgomery(src_prime_field.ToMontgomery());
  }
};

#if defined(TACHYON_GMP_BACKEND)
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpu<Config>, PrimeFieldGmp<Config>> {
  constexpr static PrimeFieldGmp<Config> Convert(
      const PrimeFieldGpu<Config>& src_prime_field) {
    return PrimeFieldGmp<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // defined(TACHYON_GMP_BACKEND)

template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpu<Config>,
                             PrimeFieldGpuDebug<Config>> {
  constexpr static PrimeFieldGpuDebug<Config> Convert(
      const PrimeFieldGpu<Config>& src_prime_field) {
    return PrimeFieldGpuDebug<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // TACHYON_CUDA

template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpuDebug<Config>, PrimeField<Config>> {
  constexpr static PrimeField<Config> Convert(
      const PrimeFieldGpuDebug<Config>& src_prime_field) {
    return PrimeField<Config>::FromMontgomery(src_prime_field.ToMontgomery());
  }
};

#if defined(TACHYON_GMP_BACKEND)
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpuDebug<Config>,
                             PrimeFieldGmp<Config>> {
  constexpr static PrimeFieldGmp<Config> Convert(
      const PrimeFieldGpuDebug<Config>& src_prime_field) {
    return PrimeFieldGmp<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // defined(TACHYON_GMP_BACKEND)

#if TACHYON_CUDA
template <typename Config>
struct PrimeFieldConversions<PrimeFieldGpuDebug<Config>,
                             PrimeFieldGpu<Config>> {
  constexpr static PrimeFieldGpu<Config> Convert(
      const PrimeFieldGpuDebug<Config>& src_prime_field) {
    return PrimeFieldGpu<Config>::FromMontgomery(
        src_prime_field.ToMontgomery());
  }
};
#endif  // TACHYON_CUDA

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_CONVERSIONS_H_

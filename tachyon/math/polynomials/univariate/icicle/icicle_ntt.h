#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_H_

#include <memory>
#include <vector>

#include "third_party/icicle/include/ntt/ntt.cu.h"

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

template <class F>
struct IsIcicleNTTSupportedImpl {
  constexpr static bool value = false;
};

template <>
struct IsIcicleNTTSupportedImpl<BabyBear> {
  constexpr static bool value = true;
};

template <>
struct IsIcicleNTTSupportedImpl<bls12_381::Fr> {
  constexpr static bool value = true;
};

template <>
struct IsIcicleNTTSupportedImpl<bn254::Fr> {
  constexpr static bool value = true;
};

template <typename F>
constexpr bool IsIcicleNTTSupported = IsIcicleNTTSupportedImpl<F>::value;

struct TACHYON_EXPORT IcicleNTTOptions {
  bool fast_twiddles_mode = true;
  int batch_size = 1;
  bool columns_batch = false;
  ::ntt::Ordering ordering = ::ntt::Ordering::kNN;
  bool are_inputs_on_device = false;
  bool are_outputs_on_device = false;
  bool is_async = false;
};

template <typename F>
class IcicleNTT {
 public:
  using BigInt = typename F::BigIntTy;

  IcicleNTT(gpuMemPool_t mem_pool, gpuStream_t stream)
      : mem_pool_(mem_pool), stream_(stream) {
    VLOG(1) << "IcicleNTT is created";
  }
  IcicleNTT(const IcicleNTT& other) = delete;
  IcicleNTT& operator=(const IcicleNTT& other) = delete;
  ~IcicleNTT() { CHECK(Release()); }

  [[nodiscard]] bool Init(const F& group_gen, const IcicleNTTOptions& options =
                                                  IcicleNTTOptions()) {
    NOTIMPLEMENTED();
    return true;
  }

  template <size_t MaxDegree>
  [[nodiscard]] bool FFT(::ntt::NttAlgorithm algorithm, const BigInt& coset,
                         UnivariateEvaluations<F, MaxDegree>& evals) const {
    const std::vector<F>& evaluations = evals.evaluations();
    return Run(algorithm, coset, const_cast<F*>(evaluations.data()),
               evaluations.size(), ::ntt::NTTDir::kForward);
  }

  template <size_t MaxDegree>
  [[nodiscard]] bool IFFT(::ntt::NttAlgorithm algorithm, const BigInt& coset,
                          UnivariateDensePolynomial<F, MaxDegree>& poly) const {
    const std::vector<F>& coefficients = poly.coefficients().coefficients();
    return Run(algorithm, coset, const_cast<F*>(coefficients.data()),
               coefficients.size(), ::ntt::NTTDir::kInverse);
  }

 private:
  [[nodiscard]] bool Run(::ntt::NttAlgorithm algorithm, const BigInt& coset,
                         F* inout, int size, ::ntt::NTTDir dir) const {
    NOTIMPLEMENTED();
    return false;
  }
  [[nodiscard]] bool Release() {
    NOTIMPLEMENTED();
    return true;
  }

  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  mutable std::unique_ptr<::ntt::NTTConfig<F>> config_;
};

template <>
TACHYON_EXPORT bool IcicleNTT<BabyBear>::Init(const BabyBear& group_gen,
                                              const IcicleNTTOptions& options);

template <>
TACHYON_EXPORT bool IcicleNTT<BabyBear>::Run(::ntt::NttAlgorithm algorithm,
                                             const BigInt& coset,
                                             BabyBear* inout, int size,
                                             ::ntt::NTTDir dir) const;

template <>
TACHYON_EXPORT bool IcicleNTT<BabyBear>::Release();

template <>
TACHYON_EXPORT bool IcicleNTT<bls12_381::Fr>::Init(
    const bls12_381::Fr& group_gen, const IcicleNTTOptions& options);

template <>
TACHYON_EXPORT bool IcicleNTT<bls12_381::Fr>::Run(::ntt::NttAlgorithm algorithm,
                                                  const BigInt& coset,
                                                  bls12_381::Fr* inout,
                                                  int size,
                                                  ::ntt::NTTDir dir) const;

template <>
TACHYON_EXPORT bool IcicleNTT<bls12_381::Fr>::Release();

template <>
TACHYON_EXPORT bool IcicleNTT<bn254::Fr>::Init(const bn254::Fr& group_gen,
                                               const IcicleNTTOptions& options);

template <>
TACHYON_EXPORT bool IcicleNTT<bn254::Fr>::Run(::ntt::NttAlgorithm algorithm,
                                              const BigInt& coset,
                                              bn254::Fr* inout, int size,
                                              ::ntt::NTTDir dir) const;

template <>
TACHYON_EXPORT bool IcicleNTT<bn254::Fr>::Release();

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_H_

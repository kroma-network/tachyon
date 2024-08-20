#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_

#include <memory>

#include "absl/types/span.h"
#include "third_party/icicle/include/hash/hash_config.h"

#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_vendor.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::crypto {

template <class F>
struct IsIciclePoseidon2SupportedImpl {
  constexpr static bool value = false;
};

template <>
struct IsIciclePoseidon2SupportedImpl<math::BabyBear> {
  constexpr static bool value = true;
};

template <>
struct IsIciclePoseidon2SupportedImpl<math::bn254::Fr> {
  constexpr static bool value = true;
};

template <typename F>
constexpr bool IsIciclePoseidon2Supported =
    IsIciclePoseidon2SupportedImpl<F>::value;

struct TACHYON_EXPORT IciclePoseidon2Options {
  bool are_inputs_on_device = false;
  bool are_outputs_on_device = false;
  bool is_async = false;
};

template <typename F>
class IciclePoseidon2 {
 public:
  IciclePoseidon2(
      gpuMemPool_t mem_pool, gpuStream_t stream,
      const IciclePoseidon2Options& options = IciclePoseidon2Options())
      : mem_pool_(mem_pool), stream_(stream) {
    ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::hash::HashConfig{
        ctx,
        options.are_inputs_on_device,
        options.are_outputs_on_device,
        options.is_async,
    });
    VLOG(1) << "IciclePoseidon2 is created";
  }
  IciclePoseidon2(const IciclePoseidon2& other) = delete;
  IciclePoseidon2& operator=(const IciclePoseidon2& other) = delete;
  ~IciclePoseidon2() { CHECK(Delete()); }

  void* impl() { return impl_; }
  const void* impl() const { return impl_; }

  [[nodiscard]] bool Create(unsigned int rate, unsigned int width,
                            unsigned int alpha, unsigned int external_rounds,
                            unsigned int internal_rounds,
                            Poseidon2Vendor external_matrix_vendor,
                            Poseidon2Vendor internal_matrix_vendor,
                            absl::Span<const F> round_constants,
                            absl::Span<const F> internal_matrix_diag);

  [[nodiscard]] bool Load(unsigned int rate, unsigned int width,
                          Poseidon2Vendor external_matrix_vendor,
                          Poseidon2Vendor internal_matrix_vendor);

  [[nodiscard]] bool Hash(unsigned int rate, absl::Span<const F> inputs,
                          absl::Span<F> outputs);

  [[nodiscard]] bool Delete();

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  void* impl_ = nullptr;
  std::unique_ptr<::hash::HashConfig> config_;
};

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Create(
    unsigned int rate, unsigned int width, unsigned int alpha,
    unsigned int external_rounds, unsigned int internal_rounds,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor,
    absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Load(
    unsigned int rate, unsigned int width,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Hash(
    unsigned int rate, absl::Span<const math::BabyBear> inputs,
    absl::Span<math::BabyBear> outputs);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Delete();

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Create(
    unsigned int rate, unsigned int width, unsigned int alpha,
    unsigned int external_rounds, unsigned int internal_rounds,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor,
    absl::Span<const math::bn254::Fr> round_constants,
    absl::Span<const math::bn254::Fr> internal_matrix_diag);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Load(
    unsigned int rate, unsigned int width,
    Poseidon2Vendor external_matrix_vendor,
    Poseidon2Vendor internal_matrix_vendor);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Hash(
    unsigned int rate, absl::Span<const math::bn254::Fr> inputs,
    absl::Span<math::bn254::Fr> outputs);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Delete();

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_

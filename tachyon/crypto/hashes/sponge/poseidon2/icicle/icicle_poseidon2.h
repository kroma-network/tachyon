#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_

#include "absl/types/span.h"
#include "third_party/icicle/include/hash/hash_config.h"

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::crypto {

enum class Vendor { kHorizen, kPlonky3 };

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

  [[nodiscard]] bool Create(unsigned int width, unsigned int rate,
                            unsigned int alpha, unsigned int internal_rounds,
                            unsigned int external_rounds,
                            absl::Span<const F> round_constants,
                            absl::Span<const F> internal_matrix_diag,
                            Vendor type);

  [[nodiscard]] bool Load(unsigned int width, unsigned int rate, Vendor type);

  [[nodiscard]] bool Hash(absl::Span<const F> inputs, F* output,
                          unsigned int number_of_states,
                          unsigned int input_block_len,
                          unsigned int output_len);

  [[nodiscard]] bool Delete() {
    if (poseidon_ != nullptr) {
      poseidon_ = nullptr;
      VLOG(1) << "Poseidon2 instance deleted";
    }
    return true;
  }

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  void* poseidon_ = nullptr;
  std::unique_ptr<::hash::HashConfig> config_;
};

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Create(
    unsigned int width, unsigned int rate, unsigned int alpha,
    unsigned int internal_rounds, unsigned int external_rounds,
    absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag, Vendor type);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Load(unsigned int width,
                                                          unsigned int rate,
                                                          Vendor type);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Hash(
    absl::Span<const math::BabyBear> inputs, math::BabyBear* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::BabyBear>::Delete();

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Create(
    unsigned int width, unsigned int rate, unsigned int alpha,
    unsigned int internal_rounds, unsigned int external_rounds,
    absl::Span<const math::bn254::Fr> round_constants,
    absl::Span<const math::bn254::Fr> internal_matrix_diag, Vendor type);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Load(unsigned int width,
                                                           unsigned int rate,
                                                           Vendor type);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Hash(
    absl::Span<const math::bn254::Fr> inputs, math::bn254::Fr* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len);

template <>
TACHYON_EXPORT bool IciclePoseidon2<math::bn254::Fr>::Delete();

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_H_

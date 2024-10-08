#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_HOLDER_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_HOLDER_H_

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"

namespace tachyon::crypto {

template <typename F>
class IciclePoseidon2Holder {
 public:
  IciclePoseidon2Holder() = default;

  const IciclePoseidon2<F>& operator*() const { return *get(); }
  IciclePoseidon2<F>& operator*() { return *get(); }

  const IciclePoseidon2<F>* operator->() const { return get(); }
  IciclePoseidon2<F>* operator->() { return get(); }

  operator bool() const { return !!icicle_; }

  const IciclePoseidon2<F>* get() const { return icicle_.get(); }
  IciclePoseidon2<F>* get() { return icicle_.get(); }

  template <size_t Rate, typename Params>
  static IciclePoseidon2Holder Create(const Poseidon2Config<Params>& config) {
    gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                             gpuMemHandleTypeNone,
                             {gpuMemLocationTypeDevice, 0}};
    device::gpu::ScopedMemPool mem_pool = device::gpu::CreateMemPool(&props);

    uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
    gpuError_t error = gpuMemPoolSetAttribute(
        mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
    CHECK_EQ(error, gpuSuccess);
    device::gpu::ScopedStream stream = device::gpu::CreateStream();

    auto icicle =
        std::make_unique<IciclePoseidon2<F>>(mem_pool.get(), stream.get());

    if constexpr (Params::kInternalMatrixVendor == Poseidon2Vendor::kPlonky3) {
      // NOTE(chokobole): It's impossible to determine if |config| was
      // constructed from |Poseidon2Config::Create()|, which accepts optimized
      // constants, so |internal_shifts| and |ark| are copied.
      std::vector<F> internal_diagonal_minus_one(Params::kWidth);
      internal_diagonal_minus_one[0] = F(F::Config::kModulus - 2);
      for (size_t i = 1; i < internal_diagonal_minus_one.size(); ++i) {
        internal_diagonal_minus_one[i] =
            F(uint32_t{1} << config.internal_shifts[i - 1]);
      }

      std::vector<F> ark;
      ark.reserve(Params::kFullRounds * Params::kWidth +
                  Params::kPartialRounds);
      Eigen::Index partial_rounds_start = Params::kFullRounds / 2;
      Eigen::Index partial_rounds_end =
          Params::kFullRounds / 2 + Params::kPartialRounds;
      for (Eigen::Index i = 0; i < config.ark.rows(); ++i) {
        if (i < partial_rounds_start || i >= partial_rounds_end) {
          for (Eigen::Index j = 0; j < config.ark.cols(); ++j) {
            ark.push_back(config.ark(i, j));
          }
        } else {
          ark.push_back(config.ark(i, 0));
        }
      }
      if (!icicle->Create(
              Rate, Params::kWidth, Params::kAlpha, Params::kFullRounds,
              Params::kPartialRounds, Params::kExternalMatrixVendor,
              Params::kInternalMatrixVendor, ark, internal_diagonal_minus_one))
        return {};
    } else {
      if (!icicle->Load(Rate, Params::kWidth, Params::kExternalMatrixVendor,
                        Params::kInternalMatrixVendor))
        return {};
    }

    return {std::move(mem_pool), std::move(stream), std::move(icicle)};
  }

  void Release() {
    icicle_.reset();
    stream_.reset();
    mem_pool_.reset();
  }

 private:
  IciclePoseidon2Holder(device::gpu::ScopedMemPool mem_pool,
                        device::gpu::ScopedStream stream,
                        std::unique_ptr<IciclePoseidon2<F>> icicle)
      : mem_pool_(std::move(mem_pool)),
        stream_(std::move(stream)),
        icicle_(std::move(icicle)) {}

  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<IciclePoseidon2<F>> icicle_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_HOLDER_H_

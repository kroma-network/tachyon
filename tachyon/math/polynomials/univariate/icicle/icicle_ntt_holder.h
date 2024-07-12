#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_HOLDER_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_HOLDER_H_

#include <limits>
#include <memory>
#include <utility>

#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt.h"

namespace tachyon::math {

template <typename F>
class IcicleNTTHolder {
 public:
  IcicleNTTHolder() = default;

  const IcicleNTT<F>& operator*() const { return *get(); }
  IcicleNTT<F>& operator*() { return *get(); }

  const IcicleNTT<F>* operator->() const { return get(); }
  IcicleNTT<F>* operator->() { return get(); }

  operator bool() const { return !!icicle_; }

  const IcicleNTT<F>* get() const { return icicle_.get(); }
  IcicleNTT<F>* get() { return icicle_.get(); }

  static IcicleNTTHolder Create() {
    gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                             gpuMemHandleTypeNone,
                             {gpuMemLocationTypeDevice, 0}};
    device::gpu::ScopedMemPool mem_pool = device::gpu::CreateMemPool(&props);

    uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
    gpuError_t error = gpuMemPoolSetAttribute(
        mem_pool.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
    CHECK_EQ(error, gpuSuccess);
    device::gpu::ScopedStream stream = device::gpu::CreateStream();

    return {std::move(mem_pool), std::move(stream),
            std::make_unique<IcicleNTT<F>>(mem_pool.get(), stream.get())};
  }

  void Release() {
    icicle_.reset();
    stream_.reset();
    mem_pool_.reset();
  }

 private:
  IcicleNTTHolder(device::gpu::ScopedMemPool mem_pool,
                  device::gpu::ScopedStream stream,
                  std::unique_ptr<IcicleNTT<F>> icicle)
      : mem_pool_(std::move(mem_pool)),
        stream_(std::move(stream)),
        icicle_(std::move(icicle)) {}

  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<IcicleNTT<F>> icicle_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_ICICLE_ICICLE_NTT_HOLDER_H_

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_HOLDER_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_HOLDER_H_

#include <limits>
#include <memory>
#include <utility>

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"

namespace tachyon::crypto {

template <typename F>
class IcicleMMCSHolder {
 public:
  IcicleMMCSHolder() = default;

  const IcicleMMCS<F>& operator*() const { return *get(); }
  IcicleMMCS<F>& operator*() { return *get(); }

  const IcicleMMCS<F>* operator->() const { return get(); }
  IcicleMMCS<F>* operator->() { return get(); }

  operator bool() const { return !!icicle_; }

  const IcicleMMCS<F>* get() const { return icicle_.get(); }
  IcicleMMCS<F>* get() { return icicle_.get(); }

  template <size_t Rate>
  static IcicleMMCSHolder Create(const void* hasher, const void* compressor) {
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
            std::make_unique<IcicleMMCS<F>>(mem_pool.get(), stream.get(),
                                            hasher, compressor, Rate)};
  }

  void Release() {
    icicle_.reset();
    stream_.reset();
    mem_pool_.reset();
  }

 private:
  IcicleMMCSHolder(device::gpu::ScopedMemPool mem_pool,
                   device::gpu::ScopedStream stream,
                   std::unique_ptr<IcicleMMCS<F>> icicle)
      : mem_pool_(std::move(mem_pool)),
        stream_(std::move(stream)),
        icicle_(std::move(icicle)) {}

  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<IcicleMMCS<F>> icicle_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_HOLDER_H_

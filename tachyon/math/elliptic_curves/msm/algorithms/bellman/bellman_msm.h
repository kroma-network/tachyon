// Copyright (c) 2022 Matter Labs
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.era-bellman-cuda and the
// LICENCE-APACHE.era-bellman-cuda file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_H_

#include <memory>

#include "tachyon/math/elliptic_curves/msm/algorithms/bellman/bellman_msm_impl.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/msm_algorithm.h"
#include "tachyon/math/elliptic_curves/msm/algorithms/pippenger/pippenger_base.h"

namespace tachyon::math {

template <typename GpuCurve>
class BellmanMSM : public PippengerBase<AffinePoint<GpuCurve>>,
                   public MSMGpuAlgorithm<GpuCurve> {
 public:
  using BaseField = typename AffinePoint<GpuCurve>::BaseField;
  using ScalarField = typename AffinePoint<GpuCurve>::ScalarField;
  using Bucket = typename PippengerBase<GpuCurve>::Bucket;
  using CpuCurve = typename GpuCurve::CpuCurve;

  BellmanMSM() : BellmanMSM(nullptr, nullptr) {}
  BellmanMSM(gpuMemPool_t mem_pool, gpuStream_t stream)
      : mem_pool_(mem_pool), stream_(stream) {
    if (!init_) {
      bellman::SetupKernels<GpuCurve>();
      init_ = true;
    }
    CHECK(mem_pool);
    CHECK(stream);
    d_results_ =
        device::gpu::GpuMemory<JacobianPoint<GpuCurve>>::MallocFromPoolAsync(
            ScalarField::Config::kModulusBits, mem_pool, stream);

    results_.reset(
        new JacobianPoint<CpuCurve>[ScalarField::Config::kModulusBits]);
  }
  BellmanMSM(const BellmanMSM& other) = delete;
  BellmanMSM& operator=(const BellmanMSM& other) = delete;

  // MSMGpuAlgorithm methods
  [[nodiscard]] bool Run(
      const device::gpu::GpuMemory<AffinePoint<GpuCurve>>& bases,
      const device::gpu::GpuMemory<ScalarField>& scalars, size_t size,
      JacobianPoint<CpuCurve>* cpu_result) override {
    bellman::ExecutionConfig<GpuCurve> config;
    config.mem_pool = mem_pool_;
    config.stream = stream_;
    config.bases = bases.get();
    config.scalars = scalars.get();
    config.results = d_results_.get();
    config.log_scalars_count = base::bits::Log2Ceiling(size);
    gpuError_t error = bellman::ExecuteAsync<GpuCurve>(config);
    if (error != gpuSuccess) return false;

    device::gpu::GpuMemcpyAsync(
        results_.get(), config.results,
        sizeof(JacobianPoint<CpuCurve>) * ScalarField::Config::kModulusBits,
        gpuMemcpyDefault, stream_);
    RETURN_AND_LOG_IF_GPU_ERROR(gpuStreamSynchronize(config.stream),
                                "Failed to gpuStreamSynchronize()");
    *cpu_result = Accumulate(results_.get());
    return true;
  }

 private:
  static JacobianPoint<CpuCurve> Accumulate(
      const JacobianPoint<CpuCurve>* results) {
    JacobianPoint<CpuCurve> ret = JacobianPoint<CpuCurve>::Zero();
    for (size_t i = 0; i < ScalarField::Config::kModulusBits; ++i) {
      size_t index = ScalarField::Config::kModulusBits - i - 1;
      JacobianPoint<CpuCurve> bucket = results[index];
      if (i == 0) {
        ret = bucket;
      } else {
        ret.DoubleInPlace();
        ret += bucket;
      }
    }
    return ret;
  }

  bool init_ = false;
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<JacobianPoint<CpuCurve>[]> results_;
  device::gpu::GpuMemory<JacobianPoint<GpuCurve>> d_results_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_BELLMAN_BELLMAN_MSM_H_

// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_

#include <memory>
#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_sponge_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_internal_matrix.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_internal_matrix.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

#if TACHYON_CUDA
#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#endif

namespace tachyon {
namespace crypto {

// Poseidon2 Sponge Hash: Absorb → Permute → Squeeze
// Absorb: Absorb elements into the sponge.
// Permute: Transform the |state| using a series of operations.
//   1. Apply ARK (addition of round constants) to |state|.
//   2. Apply S-Box (xᵅ) to |state|.
//   3. Apply external and internal matrices to |state|.
// Squeeze: Squeeze elements out of the sponge.
template <typename ExternalMatrix, typename _Params>
struct Poseidon2Sponge final
    : public PoseidonSpongeBase<Poseidon2Sponge<ExternalMatrix, _Params>> {
  using Params = _Params;
  using F = typename Params::Field;

  // Sponge Config
  Poseidon2Config<Params> config;
#if TACHYON_CUDA
  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<IciclePoseidon2<F>> poseidon2_gpu_;
#endif

  Poseidon2Sponge() = default;
  explicit Poseidon2Sponge(const Poseidon2Config<Params>& config)
      : config(config) {
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }
  explicit Poseidon2Sponge(Poseidon2Config<Params>&& config)
      : config(std::move(config)) {
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }

#if TACHYON_CUDA
  void SetupForGpu() {
    if constexpr (IsIciclePoseidon2Supported<F>) {
      if (poseidon2_gpu_) return;

      gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                               gpuMemHandleTypeNone,
                               {gpuMemLocationTypeDevice, 0}};
      mem_pool_ = device::gpu::CreateMemPool(&props);

      uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
      gpuError_t error = gpuMemPoolSetAttribute(
          mem_pool_.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
      CHECK_EQ(error, gpuSuccess);
      stream_ = device::gpu::CreateStream();

      poseidon2_gpu_.reset(
          new IciclePoseidon2<F>(mem_pool_.get(), stream_.get()));

      if (config.use_plonky3_internal_matrix) {
        math::Vector<F> internal_vector = math::Vector<F>(Params::kWidth);
        internal_vector[0] = F(F::Config::kModulus - 2);
        for (Eigen::Index i = 1; i < internal_vector.size(); ++i) {
          internal_vector[i] = F(uint32_t{1} << config.internal_shifts[i - 1]);
        }
        absl::Span<const F> internal_vector_span =
            absl::Span<const F>(internal_vector.data(), internal_vector.size());
        size_t capacity =
            Params::kFullRounds * Params::kWidth + Params::kPartialRounds;

        std::vector<F> ark_vector;
        ark_vector.reserve(capacity);
        Eigen::Index partial_rounds_start = Params::kFullRounds / 2;
        Eigen::Index partial_rounds_end =
            Params::kFullRounds / 2 + Params::kPartialRounds;
        for (Eigen::Index i = 0; i < config.ark.rows(); ++i) {
          if (i < partial_rounds_start || i >= partial_rounds_end) {
            for (Eigen::Index j = 0; j < config.ark.cols(); ++j) {
              ark_vector.push_back(config.ark(i, j));
            }
          } else {
            ark_vector.push_back(config.ark(i, 0));
          }
        }
        absl::Span<const F> ark_span =
            absl::Span<const F>(ark_vector.data(), ark_vector.size());
        if (poseidon2_gpu_->Create(Params::kWidth, Params::kRate,
                                   Params::kAlpha, Params::kPartialRounds,
                                   Params::kFullRounds, ark_span,
                                   internal_vector_span, Vendor::kPlonky3))
          return;
      } else {
        if (poseidon2_gpu_->Load(Params::kWidth, Params::kRate,
                                 Vendor::kHorizen))
          return;
      }

      LOG(ERROR) << "Failed poseidon2 gpu setup";
      poseidon2_gpu_.reset();
    }
  }
#endif

  // PoseidonSpongeBase methods
  void Permute(SpongeState<Params>& state) const {
    ApplyMixFull(state);

    size_t full_rounds_over_2 = Params::kFullRounds / 2;
    for (size_t i = 0; i < full_rounds_over_2; ++i) {
      this->ApplyARKFull(state, i);
      this->ApplySBoxFull(state);
      ApplyMixFull(state);
    }
    for (size_t i = full_rounds_over_2;
         i < full_rounds_over_2 + Params::kPartialRounds; ++i) {
      this->ApplyARKPartial(state, i);
      this->ApplySBoxPartial(state);
      ApplyMixPartial(state);
    }
    for (size_t i = full_rounds_over_2 + Params::kPartialRounds;
         i < Params::kPartialRounds + Params::kFullRounds; ++i) {
      this->ApplyARKFull(state, i);
      this->ApplySBoxFull(state);
      ApplyMixFull(state);
    }
  }

  bool operator==(const Poseidon2Sponge& other) const {
    return config == other.config;
  }
  bool operator!=(const Poseidon2Sponge& other) const {
    return !operator==(other);
  }

 private:
  void ApplyMixFull(SpongeState<Params>& state) const {
    ExternalMatrix::Apply(state.elements);
  }

  void ApplyMixPartial(SpongeState<Params>& state) const {
    using PrimeField = math::MaybeUnpack<F>;

    if constexpr (PrimeField::Config::kModulusBits <= 32) {
      if (config.use_plonky3_internal_matrix) {
        if constexpr (math::FiniteFieldTraits<F>::kIsPackedPrimeField) {
          Poseidon2Plonky3InternalMatrix<F>::Apply(
              state.elements, config.internal_diagonal_minus_one);
        } else {
          Poseidon2Plonky3InternalMatrix<F>::Apply(state.elements,
                                                   config.internal_shifts);
        }
        return;
      }
    }
    Poseidon2HorizenInternalMatrix<F>::Apply(
        state.elements, config.internal_diagonal_minus_one);
  }
};

template <typename ExternalMatrix, typename _Params>
struct CryptographicSpongeTraits<Poseidon2Sponge<ExternalMatrix, _Params>> {
  using Params = _Params;
  using F = typename Params::Field;
};

}  // namespace crypto

namespace base {
template <typename ExternalMatrix, typename Params>
class Copyable<crypto::Poseidon2Sponge<ExternalMatrix, Params>> {
 public:
  using F = typename ExternalMatrix::Field;

  static bool WriteTo(
      const crypto::Poseidon2Sponge<ExternalMatrix, Params>& poseidon,
      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config);
  }

  static bool ReadFrom(
      const ReadOnlyBuffer& buffer,
      crypto::Poseidon2Sponge<ExternalMatrix, Params>* poseidon) {
    crypto::Poseidon2Config<Params> config;
    if (!buffer.ReadMany(&config)) {
      return false;
    }

    *poseidon =
        crypto::Poseidon2Sponge<ExternalMatrix, Params>(std::move(config));
    return true;
  }

  static size_t EstimateSize(
      const crypto::Poseidon2Sponge<ExternalMatrix, Params>& poseidon) {
    return base::EstimateSize(poseidon.config);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_

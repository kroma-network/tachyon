// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

// Copyright 2022 Ethereum Foundation
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.EF and the LICENCE-APACHE.EF
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/eigen3/Eigen/LU"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/optional.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_entry.h"
#include "tachyon/crypto/hashes/sponge/poseidon/sparse_mds_matrix.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon {
namespace crypto {

// ARK(AddRoundKey) is a matrix that contains an ARC(AddRoundConstant) array in
// each row. Each constant is added to the |state| of each round of Poseidon.
// MDS(Maximum Distance Separable) is a matrix that is applied to the |state|
// for each round. It ensures that the sum of the vector's weight before and
// after the MDS is at least |state| + 1.
template <typename F>
void FindPoseidonARKAndMDS(const PoseidonGrainLFSRConfig& config,
                           size_t skip_matrices, math::Matrix<F>& ark,
                           math::Matrix<F>& mds) {
  PoseidonGrainLFSR<F> lfsr(config);
  ark = math::Matrix<F>(config.num_full_rounds + config.num_partial_rounds,
                        config.state_len);
  for (size_t i = 0; i < config.num_full_rounds + config.num_partial_rounds;
       ++i) {
    ark.row(i) = lfsr.GetFieldElementsRejectionSampling(config.state_len);
  }

  for (uint64_t i = 0; i < skip_matrices; ++i) {
    lfsr.GetFieldElementsModP(2 * config.state_len);
  }

  // a qualifying matrix must satisfy the following requirements
  // - there is no duplication among the elements in x or y
  // - there is no i and j such that x[i] + y[j] = p
  // - the resultant MDS passes all the three tests

  math::Vector<F> xs = lfsr.GetFieldElementsModP(config.state_len);
  math::Vector<F> ys = lfsr.GetFieldElementsModP(config.state_len);

  mds = math::Matrix<F>(config.state_len, config.state_len);
  for (Eigen::Index i = 0; i < mds.rows(); ++i) {
    for (Eigen::Index j = 0; j < mds.cols(); ++j) {
      mds(i, j) = unwrap((xs[i] + ys[j]).Inverse());
    }
  }
}

template <typename F>
struct PoseidonConfig : public PoseidonConfigBase<F> {
  // Maximally Distance Separating (MDS) Matrix.
  math::Matrix<F> mds;

  // See
  // https://github.com/privacy-scaling-explorations/poseidon/blob/d29f35d9744e0c58b63679a15730ff2fab470ef2/src/spec.rs#L122-L123
  math::Matrix<F> pre_sparse_mds;
  std::vector<SparseMDSMatrix<F>> sparse_mds_matrices;

  PoseidonConfig() = default;
  PoseidonConfig(const PoseidonConfigBase<F>& base, const math::Matrix<F>& mds,
                 const math::Matrix<F>& pre_sparse_mds,
                 const std::vector<SparseMDSMatrix<F>>& sparse_mds_matrices)
      : PoseidonConfigBase<F>(base),
        mds(mds),
        pre_sparse_mds(pre_sparse_mds),
        sparse_mds_matrices(sparse_mds_matrices) {}
  PoseidonConfig(PoseidonConfigBase<F>&& base, math::Matrix<F>&& mds,
                 math::Matrix<F>&& pre_sparse_mds,
                 std::vector<SparseMDSMatrix<F>>&& sparse_mds_matrices)
      : PoseidonConfigBase<F>(std::move(base)),
        mds(std::move(mds)),
        pre_sparse_mds(std::move(pre_sparse_mds)),
        sparse_mds_matrices(std::move(sparse_mds_matrices)) {}

  static PoseidonConfig CreateDefault(size_t rate, bool optimized_for_weights) {
    absl::Span<const PoseidonConfigEntry> param_set =
        optimized_for_weights ? kPoseidonOptimizedWeightsDefaultParams
                              : kPoseidonOptimizedConstraintsDefaultParams;

    auto it = base::ranges::find_if(param_set.begin(), param_set.end(),
                                    [rate](const PoseidonConfigEntry& param) {
                                      return param.rate == rate;
                                    });
    CHECK_NE(it, param_set.end());
    PoseidonConfig ret = it->template ToPoseidonConfig<F>();
    FindPoseidonARKAndMDS<F>(it->template ToPoseidonGrainLFSRConfig<F>(),
                             it->skip_matrices, ret.ark, ret.mds);
    OptimizeARK(ret.full_rounds, ret.partial_rounds, ret.mds, ret.ark);
    ComputeSparseMatrices(ret.partial_rounds, ret.mds, ret.pre_sparse_mds,
                          ret.sparse_mds_matrices);
    return ret;
  }

  constexpr static PoseidonConfig CreateCustom(size_t rate, uint64_t alpha,
                                               size_t full_rounds,
                                               size_t partial_rounds,
                                               size_t skip_matrices) {
    PoseidonConfigEntry config_entry(rate, alpha, full_rounds, partial_rounds,
                                     skip_matrices);
    PoseidonConfig ret = config_entry.ToPoseidonConfig<F>();
    FindPoseidonARKAndMDS<F>(config_entry.ToPoseidonGrainLFSRConfig<F>(),
                             skip_matrices, ret.ark, ret.mds);
    OptimizeARK(full_rounds, partial_rounds, ret.mds, ret.ark);
    ComputeSparseMatrices(partial_rounds, ret.mds, ret.pre_sparse_mds,
                          ret.sparse_mds_matrices);
    return ret;
  }

  bool IsValid() const override {
    bool ret = PoseidonConfigBase<F>::IsValid() &&
               static_cast<size_t>(mds.rows()) == this->rate + this->capacity &&
               static_cast<size_t>(mds.cols()) == this->rate + this->capacity &&
               static_cast<size_t>(pre_sparse_mds.rows()) ==
                   this->rate + this->capacity &&
               static_cast<size_t>(pre_sparse_mds.cols()) ==
                   this->rate + this->capacity &&
               sparse_mds_matrices.size() == this->partial_rounds;
    if (!ret) return false;

    for (const SparseMDSMatrix<F>& sparse_mds_matrix : sparse_mds_matrices) {
      if (sparse_mds_matrix.row().size() != mds.cols()) return false;
      if (sparse_mds_matrix.col_hat().size() != mds.rows() - 1) return false;
    }
    return true;
  }

  bool operator==(const PoseidonConfig& other) const {
    return PoseidonConfigBase<F>::operator==(other) && mds == other.mds &&
           pre_sparse_mds == other.pre_sparse_mds &&
           sparse_mds_matrices == other.sparse_mds_matrices;
  }
  bool operator!=(const PoseidonConfig& other) const {
    return !operator==(other);
  }

 private:
  constexpr static void OptimizeARK(size_t full_rounds, size_t partial_rounds,
                                    const math::Matrix<F>& mds,
                                    math::Matrix<F>& ark) {
    math::Matrix<F> mds_inverse = mds.inverse();
    math::Matrix<F> optimized_ark = {ark.rows(), ark.cols()};
    optimized_ark.row(0) = ark.row(0);
    size_t full_rounds_over_2 = full_rounds / 2;
    for (size_t i = 1; i < full_rounds_over_2; ++i) {
      optimized_ark.row(i) = mds_inverse * ark.row(i).transpose();
    }

    math::RowVector<F> acc = ark.row(full_rounds_over_2 + partial_rounds);
    for (size_t i = full_rounds_over_2; i < full_rounds_over_2 + partial_rounds;
         ++i) {
      math::RowVector<F> tmp = mds_inverse * acc.transpose();
      optimized_ark.row(full_rounds + partial_rounds - i).setZero();
      optimized_ark.row(full_rounds + partial_rounds - i)[0] = tmp[0];

      tmp[0] = F::Zero();
      acc = tmp + ark.row(full_rounds + partial_rounds - i - 1);
    }
    optimized_ark.row(full_rounds_over_2) = mds_inverse * acc.transpose();

    for (size_t i = full_rounds_over_2 + partial_rounds + 1;
         i < partial_rounds + full_rounds; ++i) {
      optimized_ark.row(i) = mds_inverse * ark.row(i).transpose();
    }
    ark = std::move(optimized_ark);
  }

  constexpr static void ComputeSparseMatrices(
      size_t partial_rounds, const math::Matrix<F>& mds,
      math::Matrix<F>& pre_sparse_mds,
      std::vector<SparseMDSMatrix<F>>& sparse_matrices) {
    math::Matrix<F> mds_transpose = mds.transpose();
    math::Matrix<F> acc = mds_transpose;
    sparse_matrices =
        base::CreateVector(partial_rounds, [&mds_transpose, &acc]() {
          math::Matrix<F> m_prime;
          SparseMDSMatrix<F> m_prime_prime;
          Factorize(acc, m_prime, m_prime_prime);
          acc = mds_transpose * m_prime;
          return m_prime_prime;
        });
    std::reverse(sparse_matrices.begin(), sparse_matrices.end());
    pre_sparse_mds = acc.transpose();
  }

  // See Section B in Supplementary Material in
  // https://eprint.iacr.org/2019/458.pdf.
  // Factorizes an MDS matrix |M| into |M'| and |M''| where |M = M' * M''|.
  // The |M'| matrix contributes to the accumulator of the process, and the
  // resulting |M''| matrix is sparse.
  constexpr static void Factorize(const math::Matrix<F>& mds,
                                  math::Matrix<F>& m_prime,
                                  SparseMDSMatrix<F>& m_prime_prime) {
    // |(r - 1 * r - 1)| sized MDS matrix called |m_hat| constructs a matrix
    // in the form |[[1 | 0], [0 | m_hat]]|, where r is rate.
    Eigen::Block<const math::Matrix<F>> m_hat =
        mds.block(1, 1, mds.rows() - 1, mds.cols() - 1);
    m_prime = math::Matrix<F>::Identity(mds.rows(), mds.cols());
    m_prime.block(1, 1, mds.rows() - 1, mds.cols() - 1) = m_hat;

    // |(r - 1)| sized vector called |w_hat| is denoted as
    // |[[m₀,₀ | m₀,ᵢ], [w_hat | identity]]|, where r is rate.
    auto w_hat = m_hat.inverse() * mds.block(1, 0, mds.rows() - 1, 1);
    math::Matrix<F> m_prime_prime_mds =
        math::Matrix<F>::Identity(mds.rows(), mds.cols());
    m_prime_prime_mds.row(0) = mds.row(0);
    m_prime_prime_mds.block(1, 0, w_hat.rows(), 1) = w_hat;
    m_prime_prime =
        SparseMDSMatrix<F>::FromMDSMatrix(m_prime_prime_mds.transpose());
  }
};

template <typename F>
PoseidonConfig<F> PoseidonConfigEntry::ToPoseidonConfig() const {
  PoseidonConfig<F> config;
  config.full_rounds = full_rounds;
  config.partial_rounds = partial_rounds;
  config.alpha = alpha;
  config.rate = rate;
  config.capacity = 1;
  return config;
}

}  // namespace crypto

namespace base {
template <typename F>
class Copyable<crypto::PoseidonConfig<F>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfig<F>& config, Buffer* buffer) {
    return Copyable<crypto::PoseidonConfigBase<F>>::WriteTo(config, buffer) &&
           buffer->WriteMany(config.mds, config.pre_sparse_mds,
                             config.sparse_mds_matrices);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfig<F>* config) {
    crypto::PoseidonConfigBase<F> base;
    math::Matrix<F> mds;
    math::Matrix<F> pre_sparse_mds;
    std::vector<crypto::SparseMDSMatrix<F>> sparse_mds_matrices;
    if (!buffer.ReadMany(&base, &mds, &pre_sparse_mds, &sparse_mds_matrices)) {
      return false;
    }

    *config = {std::move(base), std::move(mds), std::move(pre_sparse_mds),
               std::move(sparse_mds_matrices)};
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonConfig<F>& config) {
    const crypto::PoseidonConfigBase<F>& base =
        static_cast<const crypto::PoseidonConfigBase<F>&>(config);
    return base::EstimateSize(base, config.mds, config.pre_sparse_mds,
                              config.sparse_mds_matrices);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_H_

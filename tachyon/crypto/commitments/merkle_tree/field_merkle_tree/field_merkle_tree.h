// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_H_

#include <stddef.h>

#include <array>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/types/span.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon::crypto {

template <typename PackedPrimeField, size_t N>
class FieldMerkleTree {
 public:
  using PrimeField = typename PackedPrimeField::PrimeField;
  using Digest = std::array<PrimeField, N>;
  using PackedDigest = std::array<PackedPrimeField, N>;

  FieldMerkleTree() = default;

  template <typename Hasher, typename PackedHasher, typename Compressor,
            typename PackedCompressor>
  static FieldMerkleTree Build(
      const Hasher& hasher, const PackedHasher& packed_hasher,
      const Compressor& compressor, const PackedCompressor& packed_compressor,
      std::vector<math::RowMajorMatrix<PrimeField>>&& leaves) {
    CHECK(!leaves.empty());

    std::vector<const math::RowMajorMatrix<PrimeField>*> sorted_leaves =
        base::Map(leaves, [](const math::RowMajorMatrix<PrimeField>& matrix) {
          return &matrix;
        });
    // TODO(chokobole): Use https://github.com/timsort/cpp-TimSort or
    // https://github.com/sebawild/powersort for better performance.
    std::stable_sort(sorted_leaves.begin(), sorted_leaves.end(),
                     [](const math::RowMajorMatrix<PrimeField>* a,
                        const math::RowMajorMatrix<PrimeField>* b) {
                       return a->rows() > b->rows();
                     });

#if DCHECK_IS_ON()
    {
      for (size_t i = 0; i < sorted_leaves.size() - 1; ++i) {
        size_t a = static_cast<size_t>(sorted_leaves[i]->rows());
        size_t b = static_cast<size_t>(sorted_leaves[i + 1]->rows());
        CHECK(a == b || absl::bit_ceil(a) != absl::bit_ceil(b))
            << "matrix rows that round up to the same power of two must be "
               "equal";
      }
    }
#endif  // DCHECK_IS_ON()

    size_t first_layer_rows = sorted_leaves.front()->rows();
    size_t first_layer_size = 1;
    for (size_t i = 1; i < sorted_leaves.size(); ++i) {
      if (first_layer_rows == static_cast<size_t>(sorted_leaves[i]->rows())) {
        ++first_layer_size;
      } else {
        break;
      }
    }
    absl::Span<const math::RowMajorMatrix<PrimeField>*> tallest_matrices =
        absl::MakeSpan(sorted_leaves.data(), first_layer_size);
    absl::Span<const math::RowMajorMatrix<PrimeField>*> remaining_leaves =
        absl::MakeSpan(sorted_leaves.data() + first_layer_size,
                       sorted_leaves.size() - first_layer_size);

    std::vector<std::vector<Digest>> digest_layers{
        CreateFirstDigestLayer(hasher, packed_hasher, tallest_matrices)};

    while (true) {
      const std::vector<Digest>& prev_layer = digest_layers.back();
      if (prev_layer.size() == 1) break;
      size_t next_layer_rows = prev_layer.size() / 2;
      size_t next_layer_size = 0;
      for (size_t i = 0; i < remaining_leaves.size(); ++i) {
        if (absl::bit_ceil(static_cast<size_t>(remaining_leaves[i]->rows())) ==
            next_layer_rows) {
          ++next_layer_size;
        } else {
          break;
        }
      }
      absl::Span<const math::RowMajorMatrix<PrimeField>*> matrices_to_inject;
      if (next_layer_size > 0) {
        matrices_to_inject = remaining_leaves.subspan(0, next_layer_size);
        remaining_leaves.remove_prefix(next_layer_size);
      }

      digest_layers.push_back(
          CompressAndInject(hasher, packed_hasher, compressor,
                            packed_compressor, prev_layer, matrices_to_inject));
    }
    return {std::move(leaves), std::move(digest_layers)};
  }

  const std::vector<math::RowMajorMatrix<PrimeField>>& leaves() const {
    return leaves_;
  }
  const std::vector<std::vector<Digest>>& digest_layers() const {
    return digest_layers_;
  }

  const Digest& GetRoot() const { return digest_layers_.back()[0]; }

 private:
  FieldMerkleTree(std::vector<math::RowMajorMatrix<PrimeField>>&& leaves,
                  std::vector<std::vector<Digest>>&& digest_layers)
      : leaves_(std::move(leaves)), digest_layers_(std::move(digest_layers)) {}

  template <typename Hasher, typename PackedHasher>
  static std::vector<Digest> CreateFirstDigestLayer(
      const Hasher& hasher, const PackedHasher& packed_hasher,
      absl::Span<const math::RowMajorMatrix<PrimeField>*> tallest_matrices) {
    size_t max_rows = static_cast<size_t>(tallest_matrices[0]->rows());
    size_t max_rows_padded = absl::bit_ceil(max_rows);

    std::vector<Digest> ret(max_rows_padded);
    absl::Span<Digest> sub_ret = absl::MakeSpan(ret).subspan(0, max_rows);
    base::ParallelizeByChunkSize(
        sub_ret, PackedPrimeField::N,
        [&hasher, &packed_hasher, tallest_matrices](
            absl::Span<Digest> chunk, size_t chunk_offset, size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          if (chunk.size() == chunk_size) {
            std::vector<PackedPrimeField> packed_prime_fields = base::FlatMap(
                tallest_matrices,
                [start](const math::RowMajorMatrix<PrimeField>* m) {
                  return math::PackRowVertically<PackedPrimeField>(*m, start);
                });
            PackedDigest packed_digest =
                packed_hasher.Hash(packed_prime_fields);
            for (size_t i = 0; i < chunk.size(); ++i) {
              for (size_t j = 0; j < N; ++j) {
                chunk[i][j] = std::move(packed_digest[j][i]);
              }
            }
          } else {
            for (size_t i = 0; i < chunk.size(); ++i) {
              size_t row = start + i;
              chunk[i] = hasher.Hash(base::FlatMap(
                  tallest_matrices,
                  [row](const math::RowMajorMatrix<PrimeField>* m) {
                    return base::CreateVector(
                        m->cols(), [m, row](size_t i) { return (*m)(row, i); });
                  }));
            }
          }
        });
    return ret;
  }

  template <typename Hasher, typename PackedHasher, typename Compressor,
            typename PackedCompressor>
  static std::vector<Digest> CompressAndInject(
      const Hasher& hasher, const PackedHasher& packed_hasher,
      const Compressor& compressor, const PackedCompressor& packed_compressor,
      const std::vector<Digest>& prev_layer,
      absl::Span<const math::RowMajorMatrix<PrimeField>*> matrices_to_inject) {
    if (matrices_to_inject.empty())
      return Compress(compressor, packed_compressor, prev_layer);

    size_t next_rows = matrices_to_inject[0]->rows();
    size_t next_rows_padded = absl::bit_ceil(next_rows);

    std::vector<Digest> ret(next_rows_padded);
    absl::Span<Digest> sub_ret = absl::MakeSpan(ret).subspan(0, next_rows);
    base::ParallelizeByChunkSize(
        sub_ret, PackedPrimeField::N,
        [&hasher, &packed_hasher, &compressor, &packed_compressor, &prev_layer,
         matrices_to_inject](absl::Span<Digest> chunk, size_t chunk_offset,
                             size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          if (chunk.size() == chunk_size) {
            PackedDigest inputs[] = {
                base::CreateArray<N>([&prev_layer, start](size_t i) {
                  return PackedPrimeField::From(
                      [&prev_layer, start, i](size_t j) {
                        return prev_layer[2 * (start + j)][i];
                      });
                }),
                base::CreateArray<N>([&prev_layer, start](size_t i) {
                  return PackedPrimeField::From(
                      [&prev_layer, start, i](size_t j) {
                        return prev_layer[2 * (start + j) + 1][i];
                      });
                }),
            };
            inputs[0] = packed_compressor.Compress(inputs);
            std::vector<PackedPrimeField> packed_prime_fields = base::FlatMap(
                matrices_to_inject,
                [start](const math::RowMajorMatrix<PrimeField>* m) {
                  return math::PackRowVertically<PackedPrimeField>(*m, start);
                });
            inputs[1] = packed_hasher.Hash(packed_prime_fields);
            PackedDigest packed_digest = packed_compressor.Compress(inputs);
            for (size_t i = 0; i < chunk.size(); ++i) {
              for (size_t j = 0; j < N; ++j) {
                chunk[i][j] = std::move(packed_digest[j][i]);
              }
            }
          } else {
            for (size_t i = 0; i < chunk.size(); ++i) {
              Digest inputs[] = {
                  prev_layer[2 * (start + i)],
                  prev_layer[2 * (start + i) + 1],
              };
              inputs[0] = compressor.Compress(inputs);
              size_t row = start + i;
              inputs[1] = hasher.Hash(base::FlatMap(
                  matrices_to_inject,
                  [row](const math::RowMajorMatrix<PrimeField>* m) {
                    return base::CreateVector(
                        m->cols(), [m, row](size_t i) { return (*m)(row, i); });
                  }));
              chunk[i] = compressor.Compress(inputs);
            }
          }
        });

    Digest default_digest =
        base::CreateArray<N>([]() { return PrimeField::Zero(); });
    Digest inputs_with_default_digest[] = {
        default_digest,
        default_digest,
    };
    for (size_t i = next_rows; i < next_rows_padded; ++i) {
      Digest inputs[] = {
          prev_layer[2 * i],
          prev_layer[2 * i + 1],
      };
      inputs_with_default_digest[0] = compressor.Compress(inputs);
      ret[i] = compressor.Compress(inputs_with_default_digest);
    }
    return ret;
  }

  template <typename Compressor, typename PackedCompressor>
  static std::vector<Digest> Compress(const Compressor& compressor,
                                      const PackedCompressor& packed_compressor,
                                      const std::vector<Digest>& prev_layer) {
    size_t next_rows = prev_layer.size() / 2;

    std::vector<Digest> ret(next_rows);
    base::ParallelizeByChunkSize(
        ret, PackedPrimeField::N,
        [&compressor, &packed_compressor, &prev_layer](
            absl::Span<Digest> chunk, size_t chunk_offset, size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          if (chunk.size() == chunk_size) {
            PackedDigest inputs[] = {
                base::CreateArray<N>([&prev_layer, start](size_t i) {
                  return PackedPrimeField::From(
                      [&prev_layer, start, i](size_t j) {
                        return prev_layer[2 * (start + j)][i];
                      });
                }),
                base::CreateArray<N>([&prev_layer, start](size_t i) {
                  return PackedPrimeField::From(
                      [&prev_layer, start, i](size_t j) {
                        return prev_layer[2 * (start + j) + 1][i];
                      });
                }),
            };
            PackedDigest packed_digest = packed_compressor.Compress(inputs);
            for (size_t i = 0; i < chunk.size(); ++i) {
              for (size_t j = 0; j < N; ++j) {
                chunk[i][j] = std::move(packed_digest[j][i]);
              }
            }
          } else {
            for (size_t i = 0; i < chunk.size(); ++i) {
              Digest inputs[] = {
                  prev_layer[2 * (start + i)],
                  prev_layer[2 * (start + i) + 1],
              };
              chunk[i] = compressor.Compress(inputs);
            }
          }
        });
    return ret;
  }

  std::vector<math::RowMajorMatrix<PrimeField>> leaves_;
  std::vector<std::vector<Digest>> digest_layers_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_H_

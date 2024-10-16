#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_TWIDDLE_CACHE_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_TWIDDLE_CACHE_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/profiler.h"

namespace tachyon::math {

template <typename F>
class Radix2TwiddleCache {
 public:
  using PackedPrimeField =
      // NOLINTNEXTLINE(whitespace/operators)
      std::conditional_t<F::Config::kModulusBits <= 32,
                         typename PackedFieldTraits<F>::PackedField, F>;

  struct Item {
    // For small prime fields
    std::vector<F> rev_roots_vec;
    std::vector<F> rev_inv_roots_vec;
    std::vector<std::vector<PackedPrimeField>> packed_roots_vec;
    std::vector<std::vector<PackedPrimeField>> packed_inv_roots_vec;
    // For all finite fields
    std::vector<std::vector<F>> roots_vec;
    std::vector<std::vector<F>> inv_roots_vec;
    bool packed_vec_only;

    // clang-format off
    // Precompute |roots_vec| and |inv_roots_vec| for |OutInHelper()| and |InOutHelper()|.
    // Here is an example where |domain->size()| equals 32.
    // |roots_vec| = [
    //   [1],
    //   [1, ω⁸],
    //   [1, ω⁴, ω⁸, ω¹²],
    //   [1, ω², ω⁴, ω⁶, ω⁸, ω¹⁰, ω¹², ω¹⁴],
    //   [1, ω, ω², ω³, ω⁴, ω⁵, ω⁶, ω⁷, ω⁸, ω⁹, ω¹⁰, ω¹¹, ω¹², ω¹³, ω¹⁴, ω¹⁵],
    // ]
    // |inv_roots_vec| = [
    //   [1, ω⁻¹, ω⁻², ω⁻³, ω⁻⁴, ω⁻⁵, ω⁻⁶, ω⁻⁷, ω⁻⁸, ω⁻⁹, ω⁻¹⁰, ω⁻¹¹, ω⁻¹², ω⁻¹³, ω⁻¹⁴, ω⁻¹⁵],
    //   [1, ω⁻², ω⁻⁴, ω⁻⁶, ω⁻⁸, ω⁻¹⁰, ω⁻¹², ω⁻¹⁴],
    //   [1, ω⁻⁴, ω⁻⁸, ω⁻¹²],
    //   [1, ω⁻⁸],
    //   [1],
    // ]
    // clang-format on
    template <typename Domain>
    Item(const Domain* domain, bool packed_vec_only) {
      TRACE_EVENT("Utils", "Radix2TwiddleCache<F>::Item::Init");
      if (domain->log_size_of_group() == 0) return;

      this->packed_vec_only = packed_vec_only;

      roots_vec.resize(domain->log_size_of_group());
      inv_roots_vec.resize(domain->log_size_of_group());

      size_t vec_largest_size = domain->size() / 2;

      // Compute biggest vector of |roots_vec_| and |inv_roots_vec_| first.
      std::vector<F> largest =
          Domain::GetRootsOfUnity(vec_largest_size, domain->group_gen());
      std::vector<F> largest_inv =
          Domain::GetRootsOfUnity(vec_largest_size, domain->group_gen_inv());

      if constexpr (F::Config::kModulusBits <= 32) {
        TRACE_EVENT("Subtask", "PreparePackedVec");
        packed_roots_vec.resize(2);
        packed_inv_roots_vec.resize(2);
        packed_roots_vec[0].resize(vec_largest_size);
        packed_inv_roots_vec[0].resize(vec_largest_size);
        packed_roots_vec[1].resize(vec_largest_size);
        packed_inv_roots_vec[1].resize(vec_largest_size);
        rev_roots_vec = SwapBitRevElements(largest);
        rev_inv_roots_vec = SwapBitRevElements(largest_inv);
        OMP_PARALLEL_FOR(size_t i = 0; i < vec_largest_size; ++i) {
          packed_roots_vec[0][i] = PackedPrimeField::Broadcast(largest[i]);
          packed_inv_roots_vec[0][i] =
              PackedPrimeField::Broadcast(largest_inv[i]);
          packed_roots_vec[1][i] =
              PackedPrimeField::Broadcast(rev_roots_vec[i]);
          packed_inv_roots_vec[1][i] =
              PackedPrimeField::Broadcast(rev_inv_roots_vec[i]);
        }
      }

      TRACE_EVENT("Subtask", "PrepareRootsVec");

      roots_vec[domain->log_size_of_group() - 1] = std::move(largest);
      inv_roots_vec[0] = std::move(largest_inv);

      if (packed_vec_only) return;

      // Prepare space in each vector for the others.
      size_t size = domain->size() / 2;
      for (size_t i = 1; i < domain->log_size_of_group(); ++i) {
        size /= 2;
        roots_vec[domain->log_size_of_group() - i - 1].resize(size);
        inv_roots_vec[i].resize(size);
      }

      // Assign every element based on the biggest vector.
      OMP_PARALLEL_FOR(size_t i = 1; i < domain->log_size_of_group(); ++i) {
        size_t pow2_i = size_t{1} << i;
        for (size_t j = 0; j < domain->size() / (pow2_i << 1); ++j) {
          size_t k = pow2_i * j;
          roots_vec[domain->log_size_of_group() - i - 1][j] =
              roots_vec.back()[k];
          inv_roots_vec[i][j] = inv_roots_vec.front()[k];
        }
      }
    }
  };

  template <typename Domain>
  static Item* GetItem(const Domain* domain, bool packed_vec_only) {
    static base::NoDestructor<Radix2TwiddleCache> twiddle_cache;

    absl::MutexLock lock(&twiddle_cache->mutex_);
    auto it = twiddle_cache->items_.find(
        std::make_pair(domain->size(), domain->group_gen()));
    if (it == twiddle_cache->items_.end() ||
        (it->second->packed_vec_only && !packed_vec_only)) {
      it = twiddle_cache->items_.insert(
          it,
          std::make_pair(std::make_pair(domain->size(), domain->group_gen()),
                         std::make_unique<Item>(domain, packed_vec_only)));
    }
    return it->second.get();
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<std::pair<size_t, F>, std::unique_ptr<Item>> items_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_RADIX2_TWIDDLE_CACHE_H_

#ifndef TACHYON_BASE_PARALLELIZE_H_
#define TACHYON_BASE_PARALLELIZE_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/openmp_util.h"

namespace tachyon::base {

template <typename T>
using ParallelizeCallback1 = std::function<void(absl::Span<T>)>;
template <typename T>
using ParallelizeCallback2 = std::function<void(absl::Span<T>, size_t)>;
template <typename T>
using ParallelizeCallback3 = std::function<void(absl::Span<T>, size_t, size_t)>;

// Splits the |container| by |chunk_size| and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          typename T = typename SpanTy::value_type,
          size_t ArgNum = internal::GetSize<ArgList>>
void ParallelizeByChunkSize(Container& container, size_t chunk_size,
                            Callable callback) {
  if (chunk_size == 0) return;
  size_t num_chunks = (std::size(container) + chunk_size - 1) / chunk_size;
  if (num_chunks == 1) {
    SpanTy chunk(std::data(container), std::size(container));
    if constexpr (ArgNum == 1) {
      callback(chunk);
    } else if constexpr (ArgNum == 2) {
      callback(chunk, 0);
    } else {
      static_assert(ArgNum == 3);
      callback(chunk, 0, chunk_size);
    }
  } else {
    OPENMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
      size_t len = i == num_chunks - 1 ? std::size(container) - i * chunk_size
                                       : chunk_size;
      SpanTy chunk(std::data(container) + i * chunk_size, len);
      if constexpr (ArgNum == 1) {
        callback(chunk);
      } else if constexpr (ArgNum == 2) {
        callback(chunk, i);
      } else {
        static_assert(ArgNum == 3);
        callback(chunk, i, chunk_size);
      }
    }
  }
}

// Splits the |container| into threads and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable>
void Parallelize(Container& container, Callable callback,
                 std::optional<size_t> threshold = std::nullopt,
                 std::optional<size_t> num_of_threads = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold, num_of_threads);
  ParallelizeByChunkSize(container, num_elements_per_thread,
                         std::move(callback));
}

// Splits the |container| by |chunk_size| and maps each chunk using the provided
// |callback| in parallel. Each callback's return value is collected into a
// vector which is then returned.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          typename T = typename SpanTy::value_type,
          size_t ArgNum = internal::GetSize<ArgList>>
std::vector<ReturnType> ParallelizeMapByChunkSize(Container& container,
                                                  size_t chunk_size,
                                                  Callable callback) {
  if (chunk_size == 0) return {};
  size_t num_chunks = (std::size(container) + chunk_size - 1) / chunk_size;
  std::vector<ReturnType> values(num_chunks);
  if (num_chunks == 1) {
    SpanTy chunk(std::data(container), std::size(container));
    if constexpr (ArgNum == 1) {
      values[0] = callback(chunk);
    } else if constexpr (ArgNum == 2) {
      values[0] = callback(chunk, 0);
    } else {
      static_assert(ArgNum == 3);
      values[0] = callback(chunk, 0, chunk_size);
    }
    return values;
  }
  OPENMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    size_t len = i == num_chunks - 1 ? std::size(container) - i * chunk_size
                                     : chunk_size;
    SpanTy chunk(std::data(container) + i * chunk_size, len);
    if constexpr (ArgNum == 1) {
      values[i] = callback(chunk);
    } else if constexpr (ArgNum == 2) {
      values[i] = callback(chunk, i);
    } else {
      static_assert(ArgNum == 3);
      values[i] = callback(chunk, i, chunk_size);
    }
  }
  return values;
}

// Splits the |container| into threads and maps each chunk using the provided
// |callback| in parallel. The results from each callback are collected into a
// vector and returned.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable>
auto ParallelizeMap(Container& container, Callable callback,
                    std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold);
  return ParallelizeMapByChunkSize(container, num_elements_per_thread,
                                   std::move(callback));
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_PARALLELIZE_H_

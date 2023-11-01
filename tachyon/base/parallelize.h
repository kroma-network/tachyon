#ifndef TACHYON_BASE_PARALLELIZE_H_
#define TACHYON_BASE_PARALLELIZE_H_

#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/openmp_util.h"

namespace tachyon::base {

// Splits the |container| by |chunk_size| and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename ContainerTy, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          typename T = typename SpanTy::value_type,
          size_t ArgNum = internal::GetSize<ArgList>>
void ParallelizeByChunkSize(ContainerTy& container, size_t chunk_size,
                            Callable callback) {
  internal::ChunkedAdapter<ContainerTy> chunked_adapter =
      base::Chunked(container, chunk_size);
  std::vector<SpanTy> chunks =
      base::Map(chunked_adapter.begin(), chunked_adapter.end(),
                [](SpanTy chunk) { return chunk; });
  OPENMP_PARALLEL_FOR(size_t i = 0; i < chunks.size(); ++i) {
    if constexpr (ArgNum == 1) {
      callback(chunks[i]);
    } else if constexpr (ArgNum == 2) {
      callback(chunks[i], i);
    } else {
      static_assert(ArgNum == 3);
      callback(chunks[i], i, chunk_size);
    }
  }
}

// Splits the |container| into threads and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename ContainerTy, typename Callable>
void Parallelize(ContainerTy& container, Callable callback,
                 std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold);
  ParallelizeByChunkSize(container, num_elements_per_thread,
                         std::move(callback));
}

// Splits the |container| by |chunk_size| and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename ContainerTy, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          typename T = typename SpanTy::value_type,
          size_t ArgNum = internal::GetSize<ArgList>>
std::vector<ReturnType> ParallelizeMapByChunkSize(ContainerTy& container,
                                                  size_t chunk_size,
                                                  Callable callback) {
  internal::ChunkedAdapter<ContainerTy> chunked_adapter =
      base::Chunked(container, chunk_size);
  std::vector<SpanTy> chunks =
      base::Map(chunked_adapter.begin(), chunked_adapter.end(),
                [](SpanTy chunk) { return chunk; });
  std::vector<ReturnType> values;
  values.resize(chunks.size());
  OPENMP_PARALLEL_FOR(size_t i = 0; i < chunks.size(); ++i) {
    if constexpr (ArgNum == 1) {
      values[i] = callback(chunks[i]);
    } else if constexpr (ArgNum == 2) {
      values[i] = callback(chunks[i], i);
    } else {
      static_assert(ArgNum == 3);
      values[i] = callback(chunks[i], i, chunk_size);
    }
  }
  return values;
}

// Splits the |container| into threads and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename ContainerTy, typename Callable>
auto ParallelizeMap(ContainerTy& container, Callable callback,
                    std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold);
  return ParallelizeMapByChunkSize(container, num_elements_per_thread,
                                   std::move(callback));
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_PARALLELIZE_H_

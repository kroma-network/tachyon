#ifndef TACHYON_BASE_PARALLELIZE_H_
#define TACHYON_BASE_PARALLELIZE_H_

#include <optional>
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
  std::vector<absl::Span<T>> chunks =
      base::Map(chunked_adapter.begin(), chunked_adapter.end(),
                [](absl::Span<T> chunk) { return chunk; });
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
template <typename ContainerTy, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          typename T = typename SpanTy::value_type,
          size_t ArgNum = internal::GetSize<ArgList>>
void Parallelize(ContainerTy& container, Callable callback,
                 std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold);
  return ParallelizeByChunkSize(container, num_elements_per_thread, callback);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_PARALLELIZE_H_

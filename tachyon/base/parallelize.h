#ifndef TACHYON_BASE_PARALLELIZE_H_
#define TACHYON_BASE_PARALLELIZE_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/openmp_util.h"

namespace tachyon::base {
namespace internal {

template <typename Container, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          size_t ArgNum = internal::GetSize<ArgList>>
void InvokeParallelizeCallback(Container& container, size_t i,
                               size_t num_chunks, size_t chunk_size,
                               Callable callback) {
  size_t len =
      i == num_chunks - 1 ? std::size(container) - i * chunk_size : chunk_size;
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

template <typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>>
void InvokeParallelizeCallback(size_t size, size_t i, size_t num_chunks,
                               size_t chunk_size, Callable callback) {
  size_t len = i == num_chunks - 1 ? size - i * chunk_size : chunk_size;
  if constexpr (ArgNum == 1) {
    callback(len);
  } else if constexpr (ArgNum == 2) {
    callback(len, i);
  } else {
    static_assert(ArgNum == 3);
    callback(len, i, chunk_size);
  }
}

template <typename Container, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename ArgList = internal::ExtractArgs<RunType>,
          typename SpanTy = internal::GetType<0, ArgList>,
          size_t ArgNum = internal::GetSize<ArgList>>
void InvokeParallelizeCallback(Container& container, size_t i,
                               size_t num_chunks, size_t chunk_size,
                               Callable callback,
                               std::vector<ReturnType>& values) {
  size_t len =
      i == num_chunks - 1 ? std::size(container) - i * chunk_size : chunk_size;
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

template <typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename ArgList = internal::ExtractArgs<RunType>,
          size_t ArgNum = internal::GetSize<ArgList>>
void InvokeParallelizeCallback(size_t size, size_t i, size_t num_chunks,
                               size_t chunk_size, Callable callback,
                               std::vector<ReturnType>& values) {
  size_t len = i == num_chunks - 1 ? size - i * chunk_size : chunk_size;
  if constexpr (ArgNum == 1) {
    values[i] = callback(len);
  } else if constexpr (ArgNum == 2) {
    values[i] = callback(len, i);
  } else {
    static_assert(ArgNum == 3);
    values[i] = callback(len, i, chunk_size);
  }
}
}  // namespace internal

template <typename T>
using ParallelizeCallback1 = std::function<void(absl::Span<T>)>;
template <typename T>
using ParallelizeCallback2 = std::function<void(absl::Span<T>, size_t)>;
template <typename T>
using ParallelizeCallback3 = std::function<void(absl::Span<T>, size_t, size_t)>;

// Splits the |container| by |chunk_size| and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable>
void ParallelizeByChunkSize(Container& container, size_t chunk_size,
                            Callable callback) {
  if (chunk_size == 0) return;
  size_t num_chunks = (std::size(container) + chunk_size - 1) / chunk_size;
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(container, 0, num_chunks, chunk_size,
                                        callback);
    return;
  }
  OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(container, i, num_chunks, chunk_size,
                                        callback);
  }
}

// Splits the |size| by |chunk_size| and executes |callback| in parallel.
template <typename Callable>
void ParallelizeByChunkSize(size_t size, size_t chunk_size, Callable callback) {
  if (chunk_size == 0) return;
  size_t num_chunks = (size + chunk_size - 1) / chunk_size;
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(size, 0, num_chunks, chunk_size,
                                        callback);
    return;
  }
  OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(size, i, num_chunks, chunk_size,
                                        callback);
  }
}

// Splits the |container| by |chunk_size| and executes |callback| in parallel.
// Dynamically schedules tasks to ensure most efficient use of threads.
template <typename Container, typename Callable>
void DynamicParallelizeByChunkSize(Container& container, size_t chunk_size,
                                   Callable callback) {
  if (chunk_size == 0) return;
  size_t num_chunks = (std::size(container) + chunk_size - 1) / chunk_size;
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(container, 0, num_chunks, chunk_size,
                                        callback);
    return;
  }
  OMP_PARALLEL_DYNAMIC_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(container, i, num_chunks, chunk_size,
                                        callback);
  }
}

// Splits the |size| by |chunk_size| and executes |callback| in parallel.
// Dynamically schedules tasks to ensure most efficient use of threads.
template <typename Callable>
void DynamicParallelizeByChunkSize(size_t size, size_t chunk_size,
                                   Callable callback) {
  if (chunk_size == 0) return;
  size_t num_chunks = (size + chunk_size - 1) / chunk_size;
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(size, 0, num_chunks, chunk_size,
                                        callback);
    return;
  }
  OMP_PARALLEL_DYNAMIC_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(size, i, num_chunks, chunk_size,
                                        callback);
  }
}

// Splits the |container| into threads and executes |callback| in parallel.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable>
void Parallelize(Container& container, Callable callback,
                 std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread =
      GetNumElementsPerThread(container, threshold);
  ParallelizeByChunkSize(container, num_elements_per_thread,
                         std::move(callback));
}

// Splits the |size| into threads and executes |callback| in parallel.
template <typename Callable>
void Parallelize(size_t size, Callable callback,
                 std::optional<size_t> threshold = std::nullopt) {
  size_t size_per_thread = GetSizePerThread(size, threshold);
  ParallelizeByChunkSize(size, size_per_thread, std::move(callback));
}

// Splits the |container| by |chunk_size| and maps each chunk using the provided
// |callback| in parallel. Each callback's return value is collected into a
// vector which is then returned.
// See parallelize_unittest.cc for more details.
template <typename Container, typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType>
std::vector<ReturnType> ParallelizeMapByChunkSize(Container& container,
                                                  size_t chunk_size,
                                                  Callable callback) {
  if (chunk_size == 0) return {};
  size_t num_chunks = (std::size(container) + chunk_size - 1) / chunk_size;
  std::vector<ReturnType> values(num_chunks);
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(container, 0, num_chunks, chunk_size,
                                        callback, values);
    return values;
  }
  OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(container, i, num_chunks, chunk_size,
                                        callback, values);
  }
  return values;
}

// Splits the |size| by |chunk_size| and maps each chunk using the provided
// |callback| in parallel. Each callback's return value is collected into a
// vector which is then returned.
template <typename Callable,
          typename FunctorTraits = internal::MakeFunctorTraits<Callable>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType>
std::vector<ReturnType> ParallelizeMapByChunkSize(size_t size,
                                                  size_t chunk_size,
                                                  Callable callback) {
  if (chunk_size == 0) return {};
  size_t num_chunks = (size + chunk_size - 1) / chunk_size;
  std::vector<ReturnType> values(num_chunks);
  if (num_chunks == 1) {
    internal::InvokeParallelizeCallback(size, 0, num_chunks, chunk_size,
                                        callback, values);
    return values;
  }
  OMP_PARALLEL_FOR(size_t i = 0; i < num_chunks; ++i) {
    internal::InvokeParallelizeCallback(size, i, num_chunks, chunk_size,
                                        callback, values);
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

// Splits the |size| into threads and maps each chunk using the provided
// |callback| in parallel. The results from each callback are collected into a
// vector and returned.
template <typename Callable>
auto ParallelizeMap(size_t size, Callable callback,
                    std::optional<size_t> threshold = std::nullopt) {
  size_t num_elements_per_thread = GetSizePerThread(size, threshold);
  return ParallelizeMapByChunkSize(size, num_elements_per_thread,
                                   std::move(callback));
}

template <typename Container>
void ParallelizeFill(Container& container, typename Container::value_type value,
                     std::optional<size_t> threshold = std::nullopt) {
  Parallelize(
      container,
      [&value](absl::Span<typename Container::value_type> chunk) {
        for (auto& v : chunk) {
          v = value;
        }
      },
      threshold);
}

template <typename Container>
void ParallelizeResize(Container& container, size_t size,
                       std::optional<size_t> threshold = std::nullopt) {
  if (container.capacity() > size) {
    container.resize(size);
  } else {
    std::vector<typename Container::value_type> new_container(size);
    auto copy_span = absl::MakeSpan(new_container).first(container.size());
    Parallelize(
        copy_span,
        [&container](absl::Span<typename Container::value_type> chunk,
                     size_t chunk_offset, size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          for (size_t i = 0; i < chunk.size(); ++i) {
            chunk[i] = std::move(container[start + i]);
          }
        },
        threshold);
    container = std::move(new_container);
  }
}

template <typename Container>
void ParallelizeResize(Container& container, size_t size,
                       typename Container::value_type value,
                       std::optional<size_t> threshold = std::nullopt) {
  if (container.capacity() > size) {
    size_t old_size = container.size();
    container.resize(size);
    auto init_span = absl::MakeSpan(container).last(size - old_size);
    Parallelize(
        init_span,
        [&value](absl::Span<typename Container::value_type> chunk) {
          std::fill(chunk.begin(), chunk.end(), value);
        },
        threshold);
  } else {
    std::vector<typename Container::value_type> new_container(size);
    Parallelize(
        new_container,
        [&container, &value](absl::Span<typename Container::value_type> chunk,
                             size_t chunk_offset, size_t chunk_size) {
          size_t start = chunk_offset * chunk_size;
          for (size_t i = 0; i < chunk.size(); ++i) {
            chunk[i] = (start + i) < container.size()
                           ? std::move(container[start + i])
                           : value;
          }
        },
        threshold);
    container = std::move(new_container);
  }
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_PARALLELIZE_H_

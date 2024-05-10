#ifndef TACHYON_BASE_ARRAY_TO_VECTOR_H_
#define TACHYON_BASE_ARRAY_TO_VECTOR_H_

#include <stddef.h>

#include <vector>

namespace tachyon::base {

template <typename T, size_t N>
const std::vector<T> ArrayToVector(const T (&arr)[N]) {
  return std::vector<T>(std::begin(arr), std::end(arr));
}

template <typename T, size_t N, size_t M>
std::vector<std::vector<T>> Array2DToVector2D(const T (&arr)[N][M]) {
  std::vector<std::vector<T>> vec;
  vec.reserve(N);
  for (const auto& inner_array : arr) {
    vec.emplace_back(std::begin(inner_array), std::end(inner_array));
  }
  return vec;
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_ARRAY_TO_VECTOR_H_

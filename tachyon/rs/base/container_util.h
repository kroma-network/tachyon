#ifndef TACHYON_RS_BASE_CONTAINER_UTIL_H_
#define TACHYON_RS_BASE_CONTAINER_UTIL_H_

#include <vector>

#include "absl/types/span.h"
#include "rust/cxx.h"

#include "tachyon/base/functional/functor_traits.h"

namespace tachyon::rs {

template <typename T>
rust::Vec<T> ConvertCppVecToRustVec(const std::vector<T>& vec) {
  rust::Vec<T> ret;
  ret.reserve(vec.size());
  for (const T& elem : vec) {
    ret.push_back(elem);
  }
  return ret;
}

template <typename T, typename UnaryOp,
          typename FunctorTraits = base::internal::MakeFunctorTraits<UnaryOp>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType>
rust::Vec<ReturnType> ConvertCppVecToRustVec(const std::vector<T>& vec,
                                             UnaryOp&& op) {
  rust::Vec<ReturnType> ret;
  ret.reserve(vec.size());
  for (const T& elem : vec) {
    ret.push_back(op(elem));
  }
  return ret;
}

template <typename R, typename T>
rust::Slice<const R> ConvertCppVecToRustSlice(const std::vector<T>& vec) {
  return {reinterpret_cast<const R*>(vec.data()), vec.size()};
}

template <typename R, typename T>
absl::Span<R> ConvertRustSliceToCppSpan(rust::Slice<T> slice) {
  return {reinterpret_cast<R*>(slice.data()), slice.size()};
}

}  // namespace tachyon::rs

#endif  // TACHYON_RS_BASE_CONTAINER_UTIL_H_

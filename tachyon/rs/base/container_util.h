#ifndef TACHYON_RS_BASE_CONTAINER_UTIL_H_
#define TACHYON_RS_BASE_CONTAINER_UTIL_H_

#include "absl/types/span.h"
#include "rust/cxx.h"

#include "tachyon/base/functional/functor_traits.h"
#include "tachyon/base/template_util.h"

namespace tachyon::rs {

template <typename Container, typename T = base::container_value_t<Container>>
rust::Vec<T> ConvertCppContainerToRustVec(const Container& container) {
  rust::Vec<T> ret;
  ret.reserve(std::size(container));
  for (const T& elem : container) {
    ret.push_back(elem);
  }
  return ret;
}

template <typename Container, typename UnaryOp,
          typename FunctorTraits = base::internal::MakeFunctorTraits<UnaryOp>,
          typename RunType = typename FunctorTraits::RunType,
          typename ReturnType = typename FunctorTraits::ReturnType,
          typename T = base::container_value_t<Container>>
rust::Vec<ReturnType> ConvertCppContainerToRustVec(const Container& container,
                                                   UnaryOp&& op) {
  rust::Vec<ReturnType> ret;
  ret.reserve(std::size(container));
  for (const T& elem : container) {
    ret.push_back(op(elem));
  }
  return ret;
}

template <typename R, typename Container>
rust::Slice<const R> ConvertCppContainerToRustSlice(
    const Container& container) {
  return {reinterpret_cast<const R*>(std::data(container)),
          std::size(container)};
}

template <typename R, typename T>
absl::Span<R> ConvertRustSliceToCppSpan(rust::Slice<T> slice) {
  return {reinterpret_cast<R*>(slice.data()), slice.size()};
}

}  // namespace tachyon::rs

#endif  // TACHYON_RS_BASE_CONTAINER_UTIL_H_

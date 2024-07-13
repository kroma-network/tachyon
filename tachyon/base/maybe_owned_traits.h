#ifndef TACHYON_BASE_MAYBE_OWNED_TRAITS_H_
#define TACHYON_BASE_MAYBE_OWNED_TRAITS_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace tachyon::base {

template <typename T>
struct MaybeOwnedTraits {
  using Ref = T&;
  using ConstRef = const T&;
  using Ptr = T*;
  using ConstPtr = const T*;

  static T* Default() { return nullptr; }

  static void UpdatePtr(T** ptr, T& value) { *ptr = &value; }

  static void ResetPtr(T** ptr) { *ptr = nullptr; }

  static T& ToRef(T* ptr) { return *ptr; }

  static const T& ToConstRef(const T* ptr) { return *ptr; }

  static void Release(T& value) {}
};

template <typename T>
struct MaybeOwnedTraits<std::vector<T>> {
  using Ref = absl::Span<T>;
  using ConstRef = absl::Span<const T>;
  using Ptr = absl::Span<T>;
  using ConstPtr = absl::Span<const T>;

  static absl::Span<T> Default() { return {}; }

  static void UpdatePtr(absl::Span<T>* ptr, std::vector<T>& value) {
    *ptr = absl::MakeSpan(value);
  }

  static void ResetPtr(absl::Span<T>* ptr) { *ptr = absl::Span<T>(); }

  static absl::Span<T> ToRef(absl::Span<T> ptr) { return ptr; }

  static absl::Span<const T> ToConstRef(absl::Span<const T> ptr) { return ptr; }

  static void Release(std::vector<T>& value) { value.clear(); }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_MAYBE_OWNED_TRAITS_H_

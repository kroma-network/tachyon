#ifndef TACHYON_BASE_REF_H_
#define TACHYON_BASE_REF_H_

#include <utility>

#include "absl/hash/hash.h"

namespace tachyon::base {

template <typename T, bool SHALLOW = true>
class Ref {
 public:
  Ref() = default;
  explicit Ref(T* ref) : ref_(ref) {}

  const T& operator*() const { return *get(); }
  T& operator*() { return *get(); }

  const T* operator->() const { return get(); }
  T* operator->() { return get(); }

  operator bool() const { return !!ref_; }

  const T* get() const { return ref_; }
  T* get() { return ref_; }

  bool operator==(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ == other.ref_;
    } else {
      return *ref_ == *other.ref_;
    }
  }
  bool operator!=(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ != other.ref_;
    } else {
      return *ref_ != *other.ref_;
    }
  }
  bool operator<(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ < other.ref_;
    } else {
      return *ref_ < *other.ref_;
    }
  }
  bool operator<=(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ <= other.ref_;
    } else {
      return *ref_ <= *other.ref_;
    }
  }
  bool operator>(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ > other.ref_;
    } else {
      return *ref_ > *other.ref_;
    }
  }
  bool operator>=(const Ref& other) const {
    if constexpr (SHALLOW) {
      return ref_ >= other.ref_;
    } else {
      return *ref_ >= *other.ref_;
    }
  }

 private:
  // not owned
  T* ref_ = nullptr;
};

template <typename T>
using ShallowRef = Ref<T, true>;

template <typename T>
using DeepRef = Ref<T, false>;

template <typename H, typename T, bool SHALLOW>
H AbslHashValue(H h, const Ref<T, SHALLOW>& ref) {
  if constexpr (SHALLOW) {
    return H::combine(std::move(h), ref.get());
  } else {
    return H::combine(std::move(h), *ref);
  }
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_REF_H_

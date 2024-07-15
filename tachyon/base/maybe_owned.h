#ifndef TACHYON_BASE_MAYBE_OWNED_H_
#define TACHYON_BASE_MAYBE_OWNED_H_

#include <utility>

#include "tachyon/base/maybe_owned_traits.h"

namespace tachyon::base {

template <typename T>
class MaybeOwned {
 public:
  using Ref = typename MaybeOwnedTraits<T>::Ref;
  using ConstRef = typename MaybeOwnedTraits<T>::ConstRef;
  using Ptr = typename MaybeOwnedTraits<T>::Ptr;
  using ConstPtr = typename MaybeOwnedTraits<T>::ConstPtr;

  constexpr MaybeOwned() : ptr_(MaybeOwnedTraits<T>::Default()) {}
  constexpr MaybeOwned(const MaybeOwned& other)
      : value_(other.value_), owned_(other.owned_) {
    UpdatePtr(other);
  }
  constexpr MaybeOwned& operator=(const MaybeOwned& other) {
    value_ = other.value_;
    owned_ = other.owned_;
    UpdatePtr(other);
    return *this;
  }
  constexpr MaybeOwned(MaybeOwned&& other)
      : value_(std::move(other.value_)),
        owned_(std::exchange(other.owned_, false)) {
    UpdatePtr(other);
    MaybeOwnedTraits<T>::ResetPtr(&other.ptr_);
  }
  constexpr MaybeOwned& operator=(MaybeOwned&& other) {
    value_ = std::move(other.value_);
    owned_ = std::exchange(other.owned_, false);
    UpdatePtr(other);
    MaybeOwnedTraits<T>::ResetPtr(&other.ptr_);
    return *this;
  }
  // NOTE(chokobole): |explicit| keyword is explicitly removed for convenience.
  //
  // class Storage {
  //  public:
  //   Storage(const std::string& data): data_(data) {}
  //   Storage(std::string&& data): data_(std::move(data)) {}
  //  private:
  //   MaybeOwned<std::string> data_;
  // };
  // NOLINTNEXTLINE(runtime/explicit)
  constexpr MaybeOwned(const T& value) : value_(value), owned_(true) {
    MaybeOwnedTraits<T>::UpdatePtr(&ptr_, value_);
  }
  // NOLINTNEXTLINE(runtime/explicit)
  constexpr MaybeOwned(T&& value) : value_(std::move(value)), owned_(true) {
    MaybeOwnedTraits<T>::UpdatePtr(&ptr_, value_);
  }
  // NOLINTNEXTLINE(runtime/explicit)
  constexpr MaybeOwned(Ptr ptr) : ptr_(ptr) {}
  constexpr MaybeOwned& operator=(const T& value) {
    value_ = value;
    owned_ = true;
    MaybeOwnedTraits<T>::UpdatePtr(&ptr_, value_);
    return *this;
  }
  constexpr MaybeOwned& operator=(T&& value) {
    value_ = std::move(value);
    owned_ = true;
    MaybeOwnedTraits<T>::UpdatePtr(&ptr_, value_);
    return *this;
  }
  constexpr MaybeOwned& operator=(Ptr ptr) {
    MaybeOwnedTraits<T>::Release(value_);
    ptr_ = ptr;
    owned_ = false;
    return *this;
  }

  Ref operator*() { return MaybeOwnedTraits<T>::ToRef(ptr_); }
  ConstRef operator*() const { return MaybeOwnedTraits<T>::ToConstRef(ptr_); }

  Ptr operator->() { return ptr(); }
  ConstPtr operator->() const { return ptr(); }

  Ptr ptr() { return ptr_; }
  ConstPtr ptr() const { return ptr_; }

 private:
  void UpdatePtr(const MaybeOwned& other) {
    if (owned_) {
      MaybeOwnedTraits<T>::UpdatePtr(&ptr_, value_);
    } else {
      ptr_ = other.ptr_;
    }
  }

  T value_;
  Ptr ptr_;
  bool owned_ = false;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_MAYBE_OWNED_H_

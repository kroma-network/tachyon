#ifndef TACHYON_BASE_CONTAINERS_CHUNKED_ITERATOR_H_
#define TACHYON_BASE_CONTAINERS_CHUNKED_ITERATOR_H_

#include <algorithm>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"

#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/template_util.h"

namespace tachyon::base {

template <typename Iter>
class ChunkedIterator {
 public:
  constexpr static bool kIsConst = base::is_const_iterator_v<Iter>;
  using underlying_value_type =
      std::conditional_t<kIsConst, const iter_value_t<Iter>,
                         iter_value_t<Iter>>;

  using difference_type = iter_difference_t<Iter>;
  using value_type = absl::Span<underlying_value_type>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category =
      typename std::iterator_traits<Iter>::iterator_category;

  ChunkedIterator(Iter current, size_t chunk_size)
      : current_(std::move(current)),
        chunk_size_(chunk_size),
        size_(0),
        len_(0) {}
  ChunkedIterator(Iter current, size_t chunk_size, size_t size)
      : current_(std::move(current)),
        chunk_size_(chunk_size),
        size_(size),
        len_(std::min(chunk_size, size)) {}

  ChunkedIterator(const ChunkedIterator& other) = default;
  ChunkedIterator& operator=(const ChunkedIterator& other) = default;

  bool operator==(const ChunkedIterator& other) const {
    return current_ == other.current_;
  }
  bool operator!=(const ChunkedIterator& other) const {
    return !(*this == other);
  }

  // TODO(chokobole): Add more operators to support iterator category other than
  // forward_iterator.
  ChunkedIterator& operator++() {
    current_ += len_;
    offset_ += len_;
    base::CheckedNumeric<size_t> len = offset_;
    len += chunk_size_;
    size_t new_len = len.ValueOrDie();
    if (new_len > size_) {
      len_ = size_ - offset_;
    } else {
      len_ = chunk_size_;
    }
    return *this;
  }

  ChunkedIterator operator++(int) {
    ChunkedIterator iterator(*this);
    ++(*this);
    return iterator;
  }

  difference_type operator-(const ChunkedIterator& other) const {
    return (current_ - other.current_ + chunk_size_ - 1) / chunk_size_;
  }

  // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
  // mac
  const std::remove_const_t<pointer> operator->() const {
    value_ = value_type(&*current_, len_);
    return &value_;
  }
  pointer operator->() {
    return const_cast<pointer>(std::as_const(*this).operator->());
  }

  // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
  // mac
  const std::remove_const_t<reference> operator*() const {
    value_ = value_type(&*current_, len_);
    return value_;
  }
  reference operator*() {
    return const_cast<reference>(std::as_const(*this).operator*());
  }

 private:
  Iter current_;
  size_t offset_ = 0;
  size_t chunk_size_;
  size_t size_;
  size_t len_;
  mutable value_type value_;
};

template <typename T>
auto ChunkedBegin(T& t, size_t chunk_size) {
  using Iter = decltype(std::begin(std::declval<T&>()));
  return ChunkedIterator<Iter>(std::begin(t), chunk_size,
                               std::distance(std::begin(t), std::end(t)));
}

template <typename T>
auto ChunkedEnd(T& t, size_t chunk_size) {
  using Iter = decltype(std::end(std::declval<T&>()));
  return ChunkedIterator<Iter>(std::end(t), chunk_size);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_CHUNKED_ITERATOR_H_

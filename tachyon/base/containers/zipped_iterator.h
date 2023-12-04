#ifndef TACHYON_BASE_CONTAINERS_ZIPPED_ITERATOR_H_
#define TACHYON_BASE_CONTAINERS_ZIPPED_ITERATOR_H_

#include <tuple>
#include <type_traits>
#include <utility>

#include "tachyon/base/template_util.h"

namespace tachyon::base {

template <typename Iter, typename Iter2>
class ZippedIterator {
 public:
  using underlying_value_type = iter_value_t<Iter>;
  using underlying_value_type2 = iter_value_t<Iter2>;

  using difference_type = iter_difference_t<Iter>;
  using value_type = std::tuple<underlying_value_type, underlying_value_type2>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category =
      typename std::iterator_traits<Iter>::iterator_category;

  ZippedIterator(Iter current, Iter2 current2)
      : current_(std::move(current)), current2_(std::move(current2)) {}

  ZippedIterator(const ZippedIterator& other) = default;
  ZippedIterator& operator=(const ZippedIterator& other) = default;

  bool operator==(const ZippedIterator& other) const {
    return current_ == other.current_ && current2_ == other.current2_;
  }
  bool operator!=(const ZippedIterator& other) const {
    return !(*this == other);
  }

  // TODO(chokobole): Add more operators to support iterator category other than
  // forward_iterator.
  ZippedIterator& operator++() {
    ++current_;
    ++current2_;
    return *this;
  }

  ZippedIterator operator++(int) {
    ZippedIterator iterator(*this);
    ++(*this);
    return iterator;
  }

  difference_type operator-(const ZippedIterator& other) const {
    return current_ - other.current_;
  }

  // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
  // mac
  const std::remove_const_t<pointer> operator->() const {
    value_ = value_type(*current_, *current2_);
    return &value_;
  }
  pointer operator->() {
    return const_cast<pointer>(std::as_const(*this).operator->());
  }

  // NOTE(chokobole): To suppress -Werror,-Wignored-reference-qualifiers on
  // mac
  const std::remove_const_t<reference> operator*() const {
    value_ = value_type(*current_, *current2_);
    return value_;
  }
  reference operator*() {
    return const_cast<reference>(std::as_const(*this).operator*());
  }

 private:
  Iter current_;
  Iter2 current2_;
  mutable value_type value_;
};

template <typename T, typename U>
auto ZippedBegin(T& t, U& u) {
  using Iter = decltype(std::begin(std::declval<T&>()));
  using Iter2 = decltype(std::begin(std::declval<U&>()));
  return ZippedIterator<Iter, Iter2>(std::begin(t), std::begin(u));
}

template <typename T, typename U>
auto ZippedEnd(T& t, U& u) {
  using Iter = decltype(std::end(std::declval<T&>()));
  using Iter2 = decltype(std::end(std::declval<U&>()));
  return ZippedIterator<Iter, Iter2>(std::end(t), std::end(u));
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONTAINERS_ZIPPED_ITERATOR_H_

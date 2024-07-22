#ifndef TACHYON_BASE_MEMORY_REUSING_ALLOCATOR_H_
#define TACHYON_BASE_MEMORY_REUSING_ALLOCATOR_H_

#include <memory>
#include <utility>

namespace tachyon::base::memory {
template <typename T, typename A = std::allocator<T>>
class ReusingAllocator : public A {
  typedef std::allocator_traits<A> a_t;

 public:
  typedef typename a_t::size_type size_type;
  typedef typename a_t::pointer pointer;

  template <typename U>
  struct rebind {
    using other = ReusingAllocator<U, typename a_t::template rebind_alloc<U>>;
  };

  // have to store a ptr to pre-allocated memory and num of elements
  explicit ReusingAllocator(T* p = nullptr, size_type n = 0) throw() : p_(p), size_(n) {}

  ReusingAllocator(const ReusingAllocator& rhs) throw()
      : p_(rhs.p_), size_(rhs.size_) {}

  // allocate but don't initialize num elements of type T
  pointer allocate(size_type num, const void* = 0) {
    // Unless, it is the first call, and
    // it was constructed with pre-allocated memory.
    if (size_ != 0) {
      if (num == size_) {
        // Then, don't allocate; return pre-allocated mem
        size_ = 0;  // but only once
        return p_;
      } else {
        throw std::bad_alloc();
      }
    } else {
      // Allocate memory and default construct
      return new T[num]();
    }
  }

  // convert value initialization into default/new initialization
  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) {
    ::new (ptr) U();
  }

  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
  }

 private:
  pointer p_;
  size_type size_;
};

}  // namespace tachyon::base::memory

#endif  // TACHYON_BASE_MEMORY_REUSING_ALLOCATOR_H_

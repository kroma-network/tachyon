#ifndef TACHYON_ZK_BASE_REF_H_
#define TACHYON_ZK_BASE_REF_H_

namespace tachyon::zk {

template <typename T>
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

  bool operator==(const Ref& other) const { return ref_ == other.ref_; }
  bool operator!=(const Ref& other) const { return ref_ != other.ref_; }

 private:
  // not owned
  T* ref_ = nullptr;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_REF_H_

#ifndef TACHYON_BASE_BINDING_CPP_VALUE_H_
#define TACHYON_BASE_BINDING_CPP_VALUE_H_

#include "tachyon/base/logging.h"
#include "tachyon/export.h"

namespace tachyon::base {

template <typename T>
class CppRawPtr;
template <typename T>
class CppSharedPtr;
template <typename T>
class CppStackValue;
template <typename T, typename Deleter>
class CppUniquePtr;

class TACHYON_EXPORT CppValue {
 public:
  CppValue();
  virtual ~CppValue();

  virtual bool IsCppStackValue() const { return false; }

  virtual bool IsCppRawPtr() const { return false; }

  virtual bool IsCppSharedPtr() const { return false; }

  virtual bool IsCppUniquePtr() const { return false; }

  virtual void* raw_ptr() = 0;
  virtual const void* raw_ptr() const = 0;

  virtual bool is_const() const = 0;

  template <typename T>
  CppStackValue<T>* ToCppStackValue() {
    DCHECK(IsCppStackValue());
    return reinterpret_cast<CppStackValue<T>*>(this);
  }

  template <typename T>
  CppRawPtr<T>* ToCppRawPtr() {
    DCHECK(IsCppRawPtr());
    return reinterpret_cast<CppRawPtr<T>*>(this);
  }

  template <typename T>
  CppSharedPtr<T>* ToCppSharedPtr() {
    DCHECK(IsCppSharedPtr());
    return reinterpret_cast<CppSharedPtr<T>*>(this);
  }

  template <typename T, typename Deleter>
  CppUniquePtr<T, Deleter>* ToCppUniquePtr() {
    DCHECK(IsCppUniquePtr());
    return reinterpret_cast<CppUniquePtr<T, Deleter>*>(this);
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_VALUE_H_

#ifndef TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_MATCHER_H_
#define TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_MATCHER_H_

#include <memory>
#include <vector>

#include "tachyon/base/binding/cpp_constructor.h"
#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT CppConstructorMatcher {
 public:
  CppConstructorMatcher();
  CppConstructorMatcher(const CppConstructorMatcher& other) = delete;
  CppConstructorMatcher& operator=(const CppConstructorMatcher& other) = delete;
  ~CppConstructorMatcher();

  void Add(std::unique_ptr<CppConstructor> constructor) {
    ValidateAndMaybeDie(*constructor.get());
    constructors_.push_back(std::move(constructor));
  }
  size_t GetSize() const { return constructors_.size(); }

 protected:
  void ValidateAndMaybeDie(const CppConstructor& constructor) const;

  std::vector<std::unique_ptr<CppConstructor>> constructors_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BINDING_CPP_CONSTRUCTOR_MATCHER_H_

#include "tachyon/base/binding/cpp_constructor_matcher.h"

#include "tachyon/base/logging.h"

namespace tachyon::base {

CppConstructorMatcher::CppConstructorMatcher() = default;

CppConstructorMatcher::~CppConstructorMatcher() = default;

void CppConstructorMatcher::ValidateAndMaybeDie(
    const CppConstructor& constructor) const {
  for (size_t i = 0; i < constructors_.size(); ++i) {
    CppConstructor* constructor2 = constructors_[i].get();
    for (size_t default_arg_num = 0;
         default_arg_num < constructor.GetDefaultArgsNum(); ++default_arg_num) {
      for (size_t default_arg_num2 = 0;
           default_arg_num2 < constructor2->GetDefaultArgsNum();
           ++default_arg_num2) {
        if (constructor.GetArgsNum() - constructor.GetDefaultArgsNum() ==
            constructor2->GetArgsNum() - constructor2->GetDefaultArgsNum()) {
          if (constructor.GetFunctionSignature(default_arg_num) ==
              constructor2->GetFunctionSignature(default_arg_num2)) {
            LOG(DFATAL) << "You add the constructor with the same signature ("
                        << constructor.GetFunctionSignature(default_arg_num)
                        << ")";
            break;
          }
        }
      }
    }
  }
}

}  // namespace tachyon::base

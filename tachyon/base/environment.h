#ifndef TACHYON_BASE_ENVIRONMENT_H_
#define TACHYON_BASE_ENVIRONMENT_H_

#include <map>
#include <string>
#include <string_view>

#include "tachyon/export.h"

namespace tachyon {
namespace base {

// This is a wrapper class to interact with environment variables.
// NOTE: Do not call directly getenv() or setenv() inside tachyon project.
class TACHYON_EXPORT Environment {
 public:
  Environment() = delete;

  Environment(const Environment& other) = delete;
  Environment& operator=(const Environment& other) = delete;

  // Syntactic sugar for Get(variable_name, nullptr);
  static bool Has(std::string_view variable_name);
  // Returns true iff successful. This is not threadsafe.
  static bool Get(std::string_view variable_name, std::string_view* value);

  // Returns true iff successful. This is not threadsafe.
  static bool Set(std::string_view variable_name, std::string_view value);
  // Returns true iff successful. This is not threadsafe.
  static bool Unset(std::string_view variable_name);
};

using EnvironmentMap = std::map<std::string, std::string>;

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_ENVIRONMENT_H_

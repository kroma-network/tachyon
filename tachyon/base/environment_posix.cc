#include <stdlib.h>

#include "tachyon/base/environment.h"

namespace tachyon {
namespace base {

// static
bool Environment::Has(std::string_view variable_name) {
  return Get(variable_name, nullptr);
}

// static
bool Environment::Get(std::string_view variable_name, std::string_view* value) {
  const char* env_var = getenv(variable_name.data());
  if (!env_var) return false;
  if (value) *value = env_var;
  return true;
}

// static
bool Environment::Set(std::string_view variable_name, std::string_view value) {
  return setenv(variable_name.data(), value.data(), /*overwrite=*/1) == 0;
}

// static
bool Environment::Unset(std::string_view variable_name) {
  return unsetenv(variable_name.data()) == 0;
}

}  // namespace base
}  // namespace tachyon

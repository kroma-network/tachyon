#include "tachyon/base/test/scoped_environment.h"

#include "tachyon/base/environment.h"

namespace tachyon {
namespace base {

ScopedEnvironment::ScopedEnvironment(std::string_view env_name,
                                     std::string_view value)
    : env_name_(std::string(env_name)) {
  Environment::Set(env_name_, value);
}

ScopedEnvironment::~ScopedEnvironment() { Environment::Unset(env_name_); }

}  // namespace base
}  // namespace tachyon

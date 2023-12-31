#ifndef TACHYON_BASE_TEST_SCOPED_ENVIRONMENT_H_
#define TACHYON_BASE_TEST_SCOPED_ENVIRONMENT_H_

#include <string>
#include <string_view>

namespace tachyon::base {

class ScopedEnvironment {
 public:
  explicit ScopedEnvironment(std::string_view env_name, std::string_view value);
  ~ScopedEnvironment();

 private:
  std::string env_name_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_TEST_SCOPED_ENVIRONMENT_H_

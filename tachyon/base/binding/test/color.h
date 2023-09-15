#ifndef TACHYON_BASE_BINDING_TEST_COLOR_H_
#define TACHYON_BASE_BINDING_TEST_COLOR_H_

#include <string>

namespace tachyon::base::test {

enum class Color { kRed, kGreen, kBlue };

std::string ColorToString(Color c);

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_COLOR_H_

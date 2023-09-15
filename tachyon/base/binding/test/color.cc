#include "tachyon/base/binding/test/color.h"

#include "tachyon/base/logging.h"

namespace tachyon::base::test {

std::string ColorToString(Color c) {
  switch (c) {
    case Color::kRed:
      return "red";
    case Color::kGreen:
      return "green";
    case Color::kBlue:
      return "blue";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::base::test

#include <iostream>

// clang-format off
#include "tachyon/version.h"
// clang-format on

namespace tachyon {

int RealMain(int argc, char** argv) {
  std::cout << GetRuntimeFullVersionStr() << std::endl;
  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
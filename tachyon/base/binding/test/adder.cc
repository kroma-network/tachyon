#include "tachyon/base/binding/test/adder.h"

namespace tachyon::base::test {

int Adder::Add(int a, int b, int c, int d) {
  n += a + b + c + d;
  return n;
}

// static
int Adder::SAdd(int a, int b, int c, int d) { return a + b + c + d; }

}  // namespace tachyon::base::test

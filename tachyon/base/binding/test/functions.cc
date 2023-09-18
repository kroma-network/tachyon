#include "tachyon/base/binding/test/functions.h"

namespace tachyon::base::test {

namespace {
int g_global_value = 0;
}  // namespace

std::string Hello() { return "world"; }

int Sum(int a, int b) { return a + b; }

int GetGlobalValue() { return g_global_value; }

void SetGlobalValue(int v) { g_global_value = v; }

std::tuple<int, int, int> Next3(int v) {
  return std::make_tuple(v + 1, v + 2, v + 3);
}

void DoNothing() {}

}  // namespace tachyon::base::test

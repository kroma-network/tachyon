#ifndef TACHYON_BASE_BINDING_TEST_FUNCTIONS_H_
#define TACHYON_BASE_BINDING_TEST_FUNCTIONS_H_

#include <string>
#include <tuple>

namespace tachyon::base::test {

std::string Hello();

int Sum(int a, int b);

int GetGlobalValue();

void SetGlobalValue(int v);

std::tuple<int, int, int> Next3(int v);

void DoNothing();

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_FUNCTIONS_H_

#ifndef TACHYON_BASE_BINDING_TEST_ADDER_H_
#define TACHYON_BASE_BINDING_TEST_ADDER_H_

namespace tachyon::base::test {

class Adder {
 public:
  int Add(int a, int b, int c, int d);

  static int SAdd(int a, int b, int c, int d);

 private:
  int n = 0;
};

}  // namespace tachyon::base::test

#endif  // TACHYON_BASE_BINDING_TEST_ADDER_H_

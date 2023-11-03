#ifndef TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_H_
#define TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_H_

namespace tachyon::zk {

template <typename F>
class RandomFieldGenerator {
 public:
  virtual ~RandomFieldGenerator() = default;

  virtual F Generate() = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_H_

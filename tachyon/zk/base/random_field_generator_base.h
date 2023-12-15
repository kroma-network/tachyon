#ifndef TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_BASE_H_
#define TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_BASE_H_

namespace tachyon::zk {

template <typename F>
class RandomFieldGeneratorBase {
 public:
  virtual ~RandomFieldGeneratorBase() = default;

  virtual F Generate() = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_RANDOM_FIELD_GENERATOR_BASE_H_

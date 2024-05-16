#ifndef TACHYON_ZK_PLONK_EXAMPLES_POINT_H_
#define TACHYON_ZK_PLONK_EXAMPLES_POINT_H_

#include <string_view>

namespace tachyon::zk::plonk {

struct Point {
  std::string_view x;
  std::string_view y;
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXAMPLES_POINT_H_

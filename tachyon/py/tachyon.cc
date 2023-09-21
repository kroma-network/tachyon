#include "tachyon/py/math/math.h"

namespace tachyon::py {

PYBIND11_MODULE(tachyon, m) {
  m.doc() = "Bindings for tachyon.";

  math::AddMath(m);
}

}  // namespace tachyon::py

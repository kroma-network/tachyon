#ifndef TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_TRACE_H_
#define TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_TRACE_H_

#include <memory>
#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::zk::air {

template <typename F>
struct Trace {
  explicit Trace(math::RowMajorMatrix<F>& m) : main(m), preprocessed(nullptr) {}
  explicit Trace(math::RowMajorMatrix<F>&& m)
      : main(std::move(m)), preprocessed(nullptr) {}

  explicit Trace(math::RowMajorMatrix<F>& m, math::RowMajorMatrix<F>& prep)
      : main(std::move(m)),
        preprocessed(std::make_unique<math::RowMajorMatrix<F>>(prep)) {
    if (preprocessed) {
      CHECK_EQ(main.rows(), preprocessed->rows());
    }
  }
  explicit Trace(math::RowMajorMatrix<F>&& m, math::RowMajorMatrix<F>&& prep)
      : main(std::move(m)),
        preprocessed(
            std::make_unique<math::RowMajorMatrix<F>>(std::move(prep))) {
    if (preprocessed) {
      CHECK_EQ(main.rows(), preprocessed->rows());
    }
  }

  Eigen::Index rows() const { return main.rows(); }

  math::RowMajorMatrix<F> main;
  std::unique_ptr<math::RowMajorMatrix<F>> preprocessed;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_TRACE_H_

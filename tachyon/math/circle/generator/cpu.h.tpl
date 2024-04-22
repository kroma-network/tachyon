// clang-format off
#include "tachyon/base/logging.h"
#include "tachyon/math/circle/circle.h"
#include "tachyon/math/circle/circle_point.h"
#include "tachyon/math/geometry/point2.h"
#include "%{base_field_hdr}"
#include "%{scalar_field_hdr}"

namespace %{namespace} {

template <typename _BaseField, typename _ScalarField>
class %{class}CircleConfig {
 public:
  using BaseField = _BaseField;
  using BasePrimeField = %{base_prime_field};
  using ScalarField = _ScalarField;

  // TODO(chokobole): Make them constexpr.
  static Point2<BaseField> kGenerator;

  static void Init() {
%{x_init}
%{y_init}
    VLOG(1) << "%{namespace}::%{class} initialized";
  }
};

template <typename BaseField, typename ScalarField>
Point2<BaseField> %{class}CircleConfig<BaseField, ScalarField>::kGenerator;

using %{class}Circle = tachyon::math::Circle<%{class}CircleConfig<%{base_field}, %{scalar_field}>>;
using %{class}CirclePoint = tachyon::math::CirclePoint<%{class}Circle>;

}  // namespace %{namespace}
// clang-format on

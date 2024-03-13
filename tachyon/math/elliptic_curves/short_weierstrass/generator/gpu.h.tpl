// clang-format off
#include "%{base_field_header}"
#include "%{scalar_field_header}"
#include "%{config_header_path}"

namespace %{namespace} {

using %{class}CurveGpu = SWCurve<%{class}CurveConfig<%{base_field}Gpu, %{scalar_field}Gpu>>;
using %{class}AffinePointGpu = AffinePoint<%{class}CurveGpu>;
using %{class}ProjectivePointGpu = ProjectivePoint<%{class}CurveGpu>;
using %{class}JacobianPointGpu = JacobianPoint<%{class}CurveGpu>;
using %{class}PointXYZZGpu = PointXYZZ<%{class}CurveGpu>;

}  // namespace %{namespace}
// clang-format on

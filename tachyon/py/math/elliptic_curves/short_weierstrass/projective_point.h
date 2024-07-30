#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename ProjectivePoint,
          typename BaseField = typename ProjectivePoint::BaseField,
          typename ScalarField = typename ProjectivePoint::ScalarField,
          typename Curve = typename ProjectivePoint::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddProjectivePoint(py11::module& m, const std::string& name) {
  py11::class_<ProjectivePoint>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("z"))
      .def_static("zero", &ProjectivePoint::Zero)
      .def_static("generator", &ProjectivePoint::Generator)
      .def_static("random", &ProjectivePoint::Random)
      .def_property_readonly("x", &ProjectivePoint::x)
      .def_property_readonly("y", &ProjectivePoint::y)
      .def_property_readonly("z", &ProjectivePoint::z)
      .def("is_zero", &ProjectivePoint::IsZero)
      .def("is_on_curve", &ProjectivePoint::IsOnCurve)
      .def("to_string", &ProjectivePoint::ToString)
      .def("to_hex_string", &ProjectivePoint::ToHexString,
           py11::arg("pad_zero") = false)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self + py11::self)
      .def(py11::self += py11::self)
      .def(py11::self + AffinePointTy())
      .def(py11::self += AffinePointTy())
      .def(py11::self - py11::self)
      // NOTE(chokobole): See https://github.com/pybind/pybind11/issues/1893
      // .def(py11::self -= py11::self)
      .def(
          "__isub__",
          [](ProjectivePoint& lhs, const ProjectivePoint& rhs) {
            return lhs -= rhs;
          },
          py11::is_operator())
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &ProjectivePoint::Double)
      .def("double_in_place", &ProjectivePoint::DoubleInPlace)
      .def("__repr__", [name](const ProjectivePoint& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename JacobianPoint,
          typename BaseField = typename JacobianPoint::BaseField,
          typename ScalarField = typename JacobianPoint::ScalarField,
          typename Curve = typename JacobianPoint::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddJacobianPoint(py11::module& m, const std::string& name) {
  py11::class_<JacobianPoint>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("z"))
      .def_static("zero", &JacobianPoint::Zero)
      .def_static("generator", &JacobianPoint::Generator)
      .def_static("random", &JacobianPoint::Random)
      .def_property_readonly("x", &JacobianPoint::x)
      .def_property_readonly("y", &JacobianPoint::y)
      .def_property_readonly("z", &JacobianPoint::z)
      .def("is_zero", &JacobianPoint::IsZero)
      .def("is_on_curve", &JacobianPoint::IsOnCurve)
      .def("to_string", &JacobianPoint::ToString)
      .def("to_hex_string", &JacobianPoint::ToHexString,
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
          [](JacobianPoint& lhs, const JacobianPoint& rhs) {
            return lhs -= rhs;
          },
          py11::is_operator())
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &JacobianPoint::Double)
      .def("double_in_place", &JacobianPoint::DoubleInPlace)
      .def("__repr__", [name](const JacobianPoint& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

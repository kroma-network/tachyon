#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename JacobianPointTy,
          typename BaseField = typename JacobianPointTy::BaseField,
          typename ScalarField = typename JacobianPointTy::ScalarField,
          typename Curve = typename JacobianPointTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddJacobianPoint(py11::module& m, const std::string& name) {
  py11::class_<JacobianPointTy>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("z"))
      .def_static("zero", &JacobianPointTy::Zero)
      .def_static("generator", &JacobianPointTy::Generator)
      .def_static("random", &JacobianPointTy::Random)
      .def_property_readonly("x", &JacobianPointTy::x)
      .def_property_readonly("y", &JacobianPointTy::y)
      .def_property_readonly("z", &JacobianPointTy::z)
      .def("is_zero", &JacobianPointTy::IsZero)
      .def("is_on_curve", &JacobianPointTy::IsOnCurve)
      .def("to_string", &JacobianPointTy::ToString)
      .def("to_hex_string", &JacobianPointTy::ToHexString)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self + py11::self)
      .def(py11::self += py11::self)
      .def(py11::self + AffinePointTy())
      .def(py11::self += AffinePointTy())
      .def(py11::self - py11::self)
      .def(py11::self -= py11::self)
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &JacobianPointTy::Double)
      .def("double_in_place", &JacobianPointTy::DoubleInPlace)
      .def("__repr__", [name](const JacobianPointTy& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

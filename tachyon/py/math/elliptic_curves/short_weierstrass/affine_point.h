#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename AffinePoint,
          typename BaseField = typename AffinePoint::BaseField,
          typename ScalarField = typename AffinePoint::ScalarField>
void AddAffinePoint(py11::module& m, const std::string& name) {
  py11::class_<AffinePoint>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, bool>(),
           py11::arg("x"), py11::arg("y"), py11::arg("infinity") = false)
      .def_static("zero", &AffinePoint::Zero)
      .def_static("generator", &AffinePoint::Generator)
      .def_static("random", &AffinePoint::Random)
      .def_property_readonly("x", &AffinePoint::x)
      .def_property_readonly("y", &AffinePoint::y)
      .def_property_readonly("infinity", &AffinePoint::infinity)
      .def("is_zero", &AffinePoint::IsZero)
      .def("is_on_curve", &AffinePoint::IsOnCurve)
      .def("to_string", &AffinePoint::ToString)
      .def("to_hex_string", &AffinePoint::ToHexString,
           py11::arg("pad_zero") = false)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self + py11::self)
      .def(py11::self - py11::self)
      .def(py11::self * ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &AffinePoint::Double)
      .def("__repr__", [name](const AffinePoint& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

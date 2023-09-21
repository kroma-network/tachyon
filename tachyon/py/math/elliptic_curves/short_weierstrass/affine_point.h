#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include "pybind11/operators.h"

#include "tachyon/math/base/big_int.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename AffinePointTy,
          typename BaseField = typename AffinePointTy::BaseField,
          typename ScalarField = typename AffinePointTy::ScalarField>
void AddAffinePoint(py11::module& m, const std::string& name) {
  py11::class_<AffinePointTy>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, bool>(),
           py11::arg("x"), py11::arg("y"), py11::arg("infinity") = false)
      .def_static("zero", &AffinePointTy::Zero)
      .def_static("generator", &AffinePointTy::Generator)
      .def_static("random", &AffinePointTy::Random)
      .def_property_readonly("x", &AffinePointTy::x)
      .def_property_readonly("y", &AffinePointTy::y)
      .def_property_readonly("infinity", &AffinePointTy::infinity)
      .def("is_zero", &AffinePointTy::IsZero)
      .def("is_on_curve", &AffinePointTy::IsOnCurve)
      .def("to_string", &AffinePointTy::ToString)
      .def("to_hex_string", &AffinePointTy::ToHexString)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self + py11::self)
      .def(py11::self - py11::self)
      .def(py11::self * ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &AffinePointTy::Double)
      .def("__repr__", [name](const AffinePointTy& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_AFFINE_POINT_H_

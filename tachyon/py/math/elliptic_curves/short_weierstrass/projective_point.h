#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename ProjectivePointTy,
          typename BaseField = typename ProjectivePointTy::BaseField,
          typename ScalarField = typename ProjectivePointTy::ScalarField,
          typename Curve = typename ProjectivePointTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddProjectivePoint(py11::module& m, const std::string& name) {
  py11::class_<ProjectivePointTy>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("z"))
      .def_static("zero", &ProjectivePointTy::Zero)
      .def_static("generator", &ProjectivePointTy::Generator)
      .def_static("random", &ProjectivePointTy::Random)
      .def_property_readonly("x", &ProjectivePointTy::x)
      .def_property_readonly("y", &ProjectivePointTy::y)
      .def_property_readonly("z", &ProjectivePointTy::z)
      .def("is_zero", &ProjectivePointTy::IsZero)
      .def("is_on_curve", &ProjectivePointTy::IsOnCurve)
      .def("to_string", &ProjectivePointTy::ToString)
      .def("to_hex_string", &ProjectivePointTy::ToHexString)
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
          [](ProjectivePointTy& lhs, const ProjectivePointTy& rhs) {
            return lhs -= rhs;
          },
          py11::is_operator())
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &ProjectivePointTy::Double)
      .def("double_in_place", &ProjectivePointTy::DoubleInPlace)
      .def("__repr__", [name](const ProjectivePointTy& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_PROJECTIVE_POINT_H_

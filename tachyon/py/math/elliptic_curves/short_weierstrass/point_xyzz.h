#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename PointXYZZTy,
          typename BaseField = typename PointXYZZTy::BaseField,
          typename ScalarField = typename PointXYZZTy::ScalarField,
          typename Curve = typename PointXYZZTy::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddPointXYZZ(py11::module& m, const std::string& name) {
  py11::class_<PointXYZZTy>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&,
                      const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("zz"), py11::arg("zzz"))
      .def_static("zero", &PointXYZZTy::Zero)
      .def_static("generator", &PointXYZZTy::Generator)
      .def_static("random", &PointXYZZTy::Random)
      .def_property_readonly("x", &PointXYZZTy::x)
      .def_property_readonly("y", &PointXYZZTy::y)
      .def_property_readonly("zz", &PointXYZZTy::zz)
      .def_property_readonly("zzz", &PointXYZZTy::zzz)
      .def("is_zero", &PointXYZZTy::IsZero)
      .def("is_on_curve", &PointXYZZTy::IsOnCurve)
      .def("to_string", &PointXYZZTy::ToString)
      .def("to_hex_string", &PointXYZZTy::ToHexString)
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
          [](PointXYZZTy& lhs, const PointXYZZTy& rhs) { return lhs -= rhs; },
          py11::is_operator())
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &PointXYZZTy::Double)
      .def("double_in_place", &PointXYZZTy::DoubleInPlace)
      .def("__repr__", [name](const PointXYZZTy& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#ifndef TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/math/geometry/affine_point.h"
#include "tachyon/py/base/pybind11.h"

namespace tachyon::py::math {

template <typename PointXYZZ,
          typename BaseField = typename PointXYZZ::BaseField,
          typename ScalarField = typename PointXYZZ::ScalarField,
          typename Curve = typename PointXYZZ::Curve,
          typename AffinePointTy = tachyon::math::AffinePoint<Curve>>
void AddPointXYZZ(py11::module& m, const std::string& name) {
  py11::class_<PointXYZZ>(m, name.data())
      .def(py11::init<>())
      .def(py11::init<const BaseField&, const BaseField&, const BaseField&,
                      const BaseField&>(),
           py11::arg("x"), py11::arg("y"), py11::arg("zz"), py11::arg("zzz"))
      .def_static("zero", &PointXYZZ::Zero)
      .def_static("generator", &PointXYZZ::Generator)
      .def_static("random", &PointXYZZ::Random)
      .def_property_readonly("x", &PointXYZZ::x)
      .def_property_readonly("y", &PointXYZZ::y)
      .def_property_readonly("zz", &PointXYZZ::zz)
      .def_property_readonly("zzz", &PointXYZZ::zzz)
      .def("is_zero", &PointXYZZ::IsZero)
      .def("is_on_curve", &PointXYZZ::IsOnCurve)
      .def("to_string", &PointXYZZ::ToString)
      .def("to_hex_string", &PointXYZZ::ToHexString)
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
          [](PointXYZZ& lhs, const PointXYZZ& rhs) { return lhs -= rhs; },
          py11::is_operator())
      .def(py11::self - AffinePointTy())
      .def(py11::self -= AffinePointTy())
      .def(py11::self * ScalarField())
      .def(py11::self *= ScalarField())
      .def(ScalarField() * py11::self)
      .def(-py11::self)
      .def("double", &PointXYZZ::Double)
      .def("double_in_place", &PointXYZZ::DoubleInPlace)
      .def("__repr__", [name](const PointXYZZ& point) {
        return absl::Substitute("$0$1", name, point.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_POINT_XYZZ_H_

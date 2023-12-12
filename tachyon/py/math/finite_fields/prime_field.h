#ifndef TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/py/base/pybind11.h"
#include "tachyon/py/math/base/big_int.h"

namespace tachyon::py::math {

template <typename PrimeFieldTy, size_t N = PrimeFieldTy::N>
void AddPrimeField(py11::module& m, const std::string& name) {
  py11::class_<PrimeFieldTy>(m, name.data())
      .def(py11::init<const tachyon::math::BigInt<N>>())
      .def_static("zero", &PrimeFieldTy::Zero)
      .def_static("one", &PrimeFieldTy::One)
      .def_static("random", &PrimeFieldTy::Random)
      .def_static("from_dec_string", &PrimeFieldTy::FromDecString)
      .def_static("from_hex_string", &PrimeFieldTy::FromHexString)
      .def("is_zero", &PrimeFieldTy::IsZero)
      .def("is_one", &PrimeFieldTy::IsOne)
      .def("to_string", &PrimeFieldTy::ToString)
      .def("to_hex_string", &PrimeFieldTy::ToHexString,
           py11::arg("pad_zero") = false)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self < py11::self)
      .def(py11::self <= py11::self)
      .def(py11::self > py11::self)
      .def(py11::self >= py11::self)
      .def(py11::self + py11::self)
      .def(py11::self += py11::self)
      .def(py11::self - py11::self)
      // NOTE(chokobole): See https://github.com/pybind/pybind11/issues/1893
      // .def(py11::self -= py11::self)
      .def(
          "__isub__",
          [](PrimeFieldTy& lhs, const PrimeFieldTy& rhs) { return lhs -= rhs; },
          py11::is_operator())
      .def(py11::self * py11::self)
      .def(py11::self *= py11::self)
      .def(py11::self / py11::self)
      // NOTE(chokobole): See https://github.com/pybind/pybind11/issues/1893
      // .def(py11::self /= py11::self)
      .def(
          "__idiv__",
          [](PrimeFieldTy& lhs, const PrimeFieldTy& rhs) { return lhs /= rhs; },
          py11::is_operator())
      .def(-py11::self)
      .def("double", &PrimeFieldTy::Double)
      .def("double_in_place", &PrimeFieldTy::DoubleInPlace)
      .def("square", &PrimeFieldTy::Square)
      .def("square_in_place", &PrimeFieldTy::SquareInPlace)
      .def("__repr__", [name](const PrimeFieldTy& field) {
        return absl::Substitute("$0($1)", name, field.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_
